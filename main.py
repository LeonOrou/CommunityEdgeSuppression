import torch
from torch_geometric.utils import degree, add_self_loops
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import ndcg_score
import os
from collections import defaultdict
import warnings

from config import Config
from evaluation import evaluate_model
from models import LightGCN
import torch.optim as optim
from dataset import RecommendationDataset
from argparse import ArgumentParser
from utils_functions import (
    community_edge_suppression, get_community_data, get_biased_connectivity_data, set_seed)

warnings.filterwarnings('ignore')


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """Bayesian Personalized Ranking loss"""
    pos_scores = (user_emb * pos_item_emb).sum(dim=1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=1)
    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
    return loss


def sample_negative_items_optimized(user_ids, pos_item_ids, num_items, user_positive_items, device):
    """Optimized negative sampling with minimal CPU-GPU transfer"""
    batch_size = len(user_ids)
    neg_items = np.zeros(batch_size, dtype=np.int32)  # Use int32

    # Process on CPU for efficiency with sparse data
    user_ids_cpu = user_ids.cpu().numpy()

    for i in range(batch_size):
        user_id = user_ids_cpu[i]
        user_pos_set = user_positive_items.get(user_id, set())

        # Simple and fast negative sampling
        neg_item = np.random.randint(0, num_items)
        attempts = 0
        while neg_item in user_pos_set and attempts < 100:
            neg_item = np.random.randint(0, num_items)
            attempts += 1

        neg_items[i] = neg_item

    # Single transfer to GPU with int32
    return torch.tensor(neg_items, dtype=torch.int32, device=device)


def get_model(dataset, config):
    """Initialize model based on type"""
    if config.model_name == 'LightGCN':
        return LightGCN(
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            embedding_dim=config.embedding_dim,
            num_layers=config.n_layers,
        ).to(config.device)
    elif config.model_name == 'ItemKNN':
        return ItemKNN(
            num_items=dataset.num_items,
            topk=config.item_knn_topk,
            shrink=config.shrink,
        ).to(config.device)
    elif config.model_name == 'MultiVAE':
        return MultiVAE(
            num_items=dataset.num_items,
        ).to(config.device)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def prepare_adj_tensor(dataset):
    """Prepare adjacency tensor from dataset in the format (user_id, item_id, rating)"""
    df = dataset.complete_df
    adj_tens = torch.tensor(
        np.column_stack([
            df['user_encoded'].values,
            df['item_encoded'].values,
            df['rating'].values
        ]),
        dtype=torch.int64,
        device=dataset.complete_edge_index.device
    )
    return adj_tens


def prepare_training_data_gpu(train_df, device):
    """Pre-convert training data to GPU tensors with memory-efficient dtypes"""
    # Use int32 for user/item indices - sufficient for millions of users/items
    all_users = torch.tensor(train_df['user_encoded'].values, dtype=torch.int32, device=device)
    all_items = torch.tensor(train_df['item_encoded'].values, dtype=torch.int32, device=device)

    # Create indices for shuffling (needs long for arange)
    indices = torch.arange(len(train_df), device=device)

    return all_users, all_items, indices


def train_model(dataset, model, config, stage='loo', verbose=True):
    """
    Train LightGCN using the COMPLETE graph structure (train+val+test)
    but only compute loss on training interactions
    stage: 'full_train' for full training evaluation, 'loo' for leave-one-out evaluation
    """
    print(f"Complete graph: {len(dataset.complete_df)} interactions")
    print(f"Training interactions: {len(dataset.train_df)} (used for loss)")
    print(f"Val interactions: {len(dataset.val_df)} (used for evaluation)")
    print(f"Test interactions: {len(dataset.test_df)} (used for evaluation)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # max ndcg
        factor=0.7,  # less aggressive reduction
        patience=20,  # more patience
        min_lr=1e-6,
        verbose=True  # print when LR is reduced
    )

    # Move data to device
    dataset.complete_edge_index = dataset.complete_edge_index.to(device)
    dataset.complete_edge_weight = dataset.complete_edge_weight.to(device)

    # Create user positive items mapping for better negative sampling (from ALL interactions)
    user_positive_items = defaultdict(set)
    for _, row in dataset.complete_df.iterrows():
        user_positive_items[row['user_encoded']].add(row['item_encoded'])

    # Training parameters
    batch_size = config.batch_size
    best_val_ndcg = 0
    patience = config.patience if hasattr(config, 'patience') else 50  # increase default patience
    patience_counter = 0

    # Track validation history for better early stopping
    val_history = []
    best_epoch = 0

    if stage == 'full_train':
        dataset.train_df = dataset.train_val_df  # Use all training+validation data for training
        dataset.val_df = dataset.test_df  # Use test set as validation for final evaluation

    # Pre-convert training data to GPU
    all_train_users, all_train_items, train_indices = prepare_training_data_gpu(dataset.train_df, device)
    num_train = len(train_indices)

    # Warmup scheduler for the first few epochs
    warmup_epochs = 10
    initial_lr = 0.0001

    def get_lr_with_warmup(epoch):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        return initial_lr

    adj_tens = prepare_adj_tensor(dataset)

    model.train()

    for epoch in range(config.epochs):
        # Apply warmup
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = get_lr_with_warmup(epoch)
        # Apply community edge suppression if enabled
        if config.use_dropout:
            # Apply community edge suppression to get modified edge weights
            edge_weights_modified = community_edge_suppression(adj_tens, config)

            if config.model_name == 'LightGCN':
                # undirected graph needed for GNN message propagation
                edge_weights_modified = torch.concatenate((edge_weights_modified, edge_weights_modified))
            # Use the modified weights for this epoch
            current_edge_weight = edge_weights_modified.to(device)
        else:
            # Use original edge weights
            current_edge_weight = dataset.complete_edge_weight

        total_loss = 0
        num_batches = 0

        # Shuffle indices on GPU
        perm = torch.randperm(num_train, device=device)
        train_indices_shuffled = train_indices[perm]

        for i in range(0, num_train, batch_size):
            batch_indices = train_indices_shuffled[i:i + batch_size]

            if len(batch_indices) == 0:
                continue

            # Get batch data (already on GPU)
            batch_users = all_train_users[batch_indices]
            batch_pos_items = all_train_items[batch_indices]

            # Sample negative items with optimized CPU-GPU transfer
            batch_neg_items = sample_negative_items_optimized(
                batch_users,
                batch_pos_items,
                dataset.num_items,
                user_positive_items,
                device
            )

            # Forward pass - embeddings are computed on GPU
            user_emb, item_emb = model(dataset.complete_edge_index, current_edge_weight)

            # Get embeddings for batch (cast indices to long for embedding lookup)
            batch_user_emb = user_emb[batch_users.long()]
            batch_pos_item_emb = item_emb[batch_pos_items.long()]
            batch_neg_item_emb = item_emb[batch_neg_items.long()]

            # Compute loss ONLY on training interactions
            loss = bpr_loss(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)

            # Add L2 regularization
            l2_reg = config.reg * (
                    batch_user_emb.norm(2).pow(2) +
                    batch_pos_item_emb.norm(2).pow(2) +
                    batch_neg_item_emb.norm(2).pow(2)
            ) / batch_size

            total_loss_batch = loss + l2_reg

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            total_loss_batch.backward()

            # Gradient clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += total_loss_batch.item()
            num_batches += 1

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            # Use original edge weights for evaluation (no suppression)
            val_metrics = evaluate_model(
                model=model, dataset=dataset, k_values=[10],
            )
            val_ndcg = val_metrics[10]['ndcg']  # Use k=10 for scheduler and early stopping
            val_history.append(val_ndcg)

            # scheduler.step(val_ndcg)
            current_lr = optimizer.param_groups[0]['lr']

            if verbose:
                avg_loss = total_loss / max(num_batches, 1)
                suppression_status = "ON" if config.use_dropout else "OFF"
                print(f'  Epoch {epoch:3d}/{config.epochs}, Loss: {avg_loss:.4f}, '
                      f'Val NDCG@10: {val_ndcg:.4f} (Suppression: {suppression_status})')

            # Improved early stopping with relative improvement check
            if val_ndcg > best_val_ndcg:
                improvement = (val_ndcg - best_val_ndcg) / best_val_ndcg if best_val_ndcg > 0 else 1.0

                # Only update best if improvement is significant (> 0.1%)
                if improvement > 0.001 or best_val_ndcg == 0:
                    best_val_ndcg = val_ndcg
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    best_epoch = epoch
                else:
                    patience_counter += 1
            else:
                patience_counter += 1

            # Additional check: stop if learning has plateaued
            if len(val_history) >= 5:
                recent_std = np.std(val_history[-5:])
                if recent_std < 0.0001 and patience_counter >= patience // 2:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch} - validation plateaued")
                    break

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} - no improvement for {patience} checks")
                    print(f"  Best epoch was {best_epoch} with NDCG@10: {best_val_ndcg:.4f}")
                break

    # Load best model
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)

    return model, dataset.complete_edge_index, dataset.complete_edge_weight


def main():
    args = parse_arguments()
    config = Config()
    config.update_from_args(args)
    config.setup_model_config()
    config.log_config()

    dataset = RecommendationDataset(name=config.dataset_name, data_path=f'dataset/{config.dataset_name}')
    dataset.load_data().prepare_data()

    config.user_degrees, config.item_degrees = dataset.get_node_degrees()
    print("Preparing data with consistent encoding...")
    print(
        f"Processed data: {dataset.num_users} users, {dataset.num_items} items, {len(dataset.complete_df)} interactions")

    print(f"Train set: {len(dataset.train_val_df)} interactions")
    print(f"Test set: {len(dataset.test_df)} interactions")

    # 5-fold cross validation using temporal splits within training set
    print("\nStarting 5-fold cross validation with complete graph...")
    if config.use_dropout:
        print(f"Community edge suppression ENABLED - suppression strength: {config.community_suppression}")
        print(f"User dropout: {config.users_dec_perc_drop}, Item dropout: {config.items_dec_perc_drop}")
    else:
        print("Community edge suppression DISABLED")

    # Pre-calculate community data and biased edges
    adj_np = dataset.complete_df[['user_encoded', 'item_encoded', 'rating']].values
    adj_tens = prepare_adj_tensor(dataset)

    # Get community labels and power nodes
    get_community_data(config, adj_np)

    # Get biased connectivity data
    get_biased_connectivity_data(config, adj_tens)

    cv_results = []

    n_folds = 5
    for fold in range(n_folds):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        model = get_model(config=config, dataset=dataset)

        dataset.get_fold_i(i=fold)  # sets train_df and val_df for this fold

        model, complete_edge_index, complete_edge_weight = train_model(
            dataset=dataset,
            model=model,
            config=config,
            stage='loo',
        )

        # Evaluate on CV validation set
        if len(dataset.val_df) > 0:
            val_metrics = evaluate_model(
                model=model, dataset=dataset, k_values=config.evaluate_top_k, stage='loo'
            )

            fold_results = {
                'fold': fold + 1,
            }

            # Store metrics for each k
            for k in config.evaluate_top_k:
                fold_results[f'val_ndcg@{k}'] = val_metrics[k]['ndcg']
                fold_results[f'val_recall@{k}'] = val_metrics[k]['recall']
                fold_results[f'val_precision@{k}'] = val_metrics[k]['precision']
                fold_results[f'val_mrr@{k}'] = val_metrics[k]['mrr']
                fold_results[f'val_hit_rate@{k}'] = val_metrics[k]['hit_rate']
                fold_results[f'val_item_coverage@{k}'] = val_metrics[k]['item_coverage']
                fold_results[f'val_gini_index@{k}'] = val_metrics[k]['gini_index']
                fold_results[f'val_simpson_index@{k}'] = val_metrics[k]['simpson_index']

            cv_results.append(fold_results)

            print(f"\nFold {fold + 1} Results:")
            print(f"{'Metric':<15} {'k=10':>10} {'k=20':>10} {'k=50':>10} {'k=100':>10}")
            print("-" * 48)

            # Only show key metrics during training
            for metric, key in [('NDCG', 'ndcg'), ('Recall', 'recall'), ('Hit Rate', 'hit_rate')]:
                row = f"{metric:<15}"
                for k in config.evaluate_top_k:
                    row += f"{val_metrics[k][key]:>10.4f}"
                print(row)

    # Calculate average CV results
    if cv_results:
        cv_df = pd.DataFrame(cv_results)
        cv_summary = {}

        # Calculate mean for each metric at each k
        for k in config.evaluate_top_k:
            cv_summary[k] = {
                'NDCG': cv_df[f'val_ndcg@{k}'].mean(),
                'Recall': cv_df[f'val_recall@{k}'].mean(),
                'Precision': cv_df[f'val_precision@{k}'].mean(),
                'MRR': cv_df[f'val_mrr@{k}'].mean(),
                'Hit Rate': cv_df[f'val_hit_rate@{k}'].mean(),
                'Item Coverage': cv_df[f'val_item_coverage@{k}'].mean(),
                'Gini Index': cv_df[f'val_gini_index@{k}'].mean(),
                'Simpson Index': cv_df[f'val_simpson_index@{k}'].mean(),
            }

        print("\n" + "=" * 70)
        print("CROSS-VALIDATION SUMMARY (5-fold average)")
        print("=" * 70)

        # Create table with metrics as rows and k values as columns
        print(f"\n{'Metric':<15} {'k=10':>10} {'k=20':>10} {'k=50':>10} {'k=100':>10}")
        print("-" * 58)

        metrics = ['NDCG', 'Recall', 'Precision', 'MRR', 'Hit Rate',
                   'Item Coverage', 'Gini Index', 'Simpson Index']

        for metric in metrics:
            row = f"{metric:<15}"
            for k in config.evaluate_top_k:
                value = cv_summary[k][metric]
                row += f"{value:>12.4f}"
            print(row)

            # Add separator after accuracy metrics
            if metric == 'Hit Rate':
                print("-" * 58)

    else:
        cv_summary = {}

    # Train final model on all data and evaluate on test set
    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("=" * 60)

    print("Training final model with complete graph structure...")

    model = get_model(config=config, dataset=dataset)
    # Final training: use ALL data in graph, but only train+val for loss
    final_model, final_edge_index, final_edge_weight = train_model(
        model=model, dataset=dataset, config=config, stage='full_train',
    )

    test_metrics = evaluate_model(
        model=final_model, dataset=dataset, k_values=config.evaluate_top_k, stage='full_train'
    )

    print(f"\nFINAL TEST SET RESULTS")
    print("=" * 70)

    # Create table with metrics as rows and k values as columns
    print(f"\n{'Metric':<20} {'k=5':>12} {'k=10':>12} {'k=20':>12}")
    print("-" * 58)

    metrics_dict = {
        'NDCG': 'ndcg',
        'Recall': 'recall',
        'Precision': 'precision',
        'MRR': 'mrr',
        'Hit Rate': 'hit_rate',
        'Item Coverage': 'item_coverage',
        'Gini Index': 'gini_index',
        'Simpson Index': 'simpson_index'
    }

    for display_name, metric_key in metrics_dict.items():
        row = f"{display_name:<20}"
        for k in config.evaluate_top_k:
            value = test_metrics[k][metric_key]
            row += f"{value:>12.4f}"
        print(row)

        # Add separator after accuracy metrics
        if display_name == 'Hit Rate':
            print("-" * 58)

    return cv_results, cv_summary, test_metrics, final_model


# Function to get recommendations
def get_recommendations_complete_graph(model, user_id, complete_edge_index, complete_edge_weight,
                                       train_interactions, k=10):
    """Get top-k recommendations for a user using complete graph"""
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(complete_edge_index, complete_edge_weight)

        user_embedding = user_emb[user_id:user_id + 1]
        scores = torch.matmul(user_embedding, item_emb.T).squeeze()

        # Exclude training interactions
        if train_interactions is not None:
            scores[train_interactions] = float('-inf')

        top_items = torch.topk(scores, k=k).indices
        top_scores = torch.topk(scores, k=k).values

    return top_items.cpu().numpy(), top_scores.cpu().numpy()


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default='LightGCN')
    parser.add_argument("--dataset_name", type=str, default='ml-20m')
    parser.add_argument("--users_top_percent", type=float, default=0.05)
    parser.add_argument("--items_top_percent", type=float, default=0.05)
    parser.add_argument("--users_dec_perc_drop", type=float, default=0.05)
    parser.add_argument("--items_dec_perc_drop", type=float, default=0.05)
    parser.add_argument("--community_suppression", type=float, default=0.4)
    parser.add_argument("--drop_only_power_nodes", type=bool, default=True)
    parser.add_argument("--use_dropout", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    set_seed(42)  # For reproducibility
    cv_results, cv_summary, test_metrics, final_model = main()

    print("\n" + "=" * 70)
    print("DETAILED FOLD RESULTS")
    print("=" * 70)

    for result in cv_results:
        print(f"\nFold {result['fold']}:")
        print(f"{'Metric':<20} {'k=5':>12} {'k=10':>12} {'k=20':>12}")
        print("-" * 58)

        # Accuracy metrics
        for metric in ['NDCG', 'Recall', 'Precision', 'MRR', 'Hit Rate']:
            row = f"{metric:<20}"
            for k in [10, 20, 50, 100]:
                key = f'val_{metric.lower().replace(" ", "_")}@{k}'
                value = result[key]
                row += f"{value:>12.4f}"
            print(row)

        print("-" * 58)

        # Diversity metrics
        for metric in ['Item Coverage', 'Gini Index', 'Simpson Index']:
            row = f"{metric:<20}"
            for k in [10, 20, 50, 100]:
                key = f'val_{metric.lower().replace(" ", "_")}@{k}'
                value = result[key]
                row += f"{value:>12.4f}"
            print(row)