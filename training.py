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


def sample_negative_items(user_ids, pos_item_ids, num_items, user_positive_items=None, num_negatives=1):
    """Sample negative items for each user-positive item pair"""
    neg_items = []

    for user_id, pos_item_id in zip(user_ids, pos_item_ids):
        user_neg_items = []
        user_pos_set = user_positive_items.get(user_id, set()) if user_positive_items else set()

        for _ in range(num_negatives):
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                neg_item = np.random.randint(0, num_items)
                # Ensure negative item is not a positive item for this user
                if neg_item not in user_pos_set:
                    break
                attempts += 1
            user_neg_items.append(neg_item)
        neg_items.extend(user_neg_items)

    return torch.tensor(neg_items, dtype=torch.long)


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

    # Pre-calculate community data and biased edges
    adj_np = dataset.complete_df[['user_encoded', 'item_encoded', 'rating']].values
    adj_tens = prepare_adj_tensor(dataset)

    # Get community labels and power nodes
    get_community_data(config, adj_np)

    # Get biased connectivity data
    get_biased_connectivity_data(config, adj_tens)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # max ndcg
        factor=0.5,
        patience=10,
        min_lr=1e-5,
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
    patience = config.patience
    patience_counter = 0

    if stage == 'full_train':
        dataset.train_df = dataset.train_val_df  # Use all training+validation data for training
        dataset.val_df = dataset.test_df  # Use test set as validation for final evaluation

    model.train()

    for epoch in range(config.epochs):
        # Apply community edge suppression if enabled
        if config.use_dropout:
            # Apply community edge suppression to get modified edge weights
            edge_weights_modified = community_edge_suppression(adj_tens, config)

            # Use the modified weights for this epoch
            current_edge_weight = edge_weights_modified.to(device)
        else:
            # Use original edge weights
            current_edge_weight = dataset.complete_edge_weight

        total_loss = 0
        num_batches = 0

        # Sample training batches from TRAINING data only (for loss computation)
        train_interactions = dataset.train_df.sample(n=min(len(dataset.train_df), batch_size), replace=True)

        for i in range(0, len(train_interactions), batch_size):
            batch = train_interactions.iloc[i:i + batch_size]

            if len(batch) == 0:
                continue

            # Get batch data
            batch_users = torch.tensor(batch['user_encoded'].values, dtype=torch.long).to(device)
            batch_pos_items = torch.tensor(batch['item_encoded'].values, dtype=torch.long).to(device)

            # Improved negative sampling using complete interaction set
            batch_neg_items = sample_negative_items(
                batch['user_encoded'].values,
                batch['item_encoded'].values,
                dataset.num_items,
                user_positive_items=user_positive_items
            ).to(device)

            # Use the current edge weights (either modified or original)
            user_emb, item_emb = model(dataset.complete_edge_index, current_edge_weight)

            # Get embeddings for batch
            batch_user_emb = user_emb[batch_users]
            batch_pos_item_emb = item_emb[batch_pos_items]
            batch_neg_item_emb = item_emb[batch_neg_items]

            # Compute loss ONLY on training interactions
            loss = bpr_loss(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)

            # Add L2 regularization
            l2_reg = config.reg * (
                    batch_user_emb.norm(2).pow(2) +
                    batch_pos_item_emb.norm(2).pow(2) +
                    batch_neg_item_emb.norm(2).pow(2)
            ) / batch_size

            total_loss_batch = loss + l2_reg

            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()
            num_batches += 1

        # Validation every 5 epochs
        if epoch % 5 == 0:
            # Use original edge weights for evaluation (no suppression)
            val_metrics = evaluate_model(
                model=model, dataset=dataset, k=10,
            )
            val_ndcg = val_metrics['ndcg']

            # scheduler.step(val_ndcg)

            if verbose and epoch % 10 == 0:
                avg_loss = total_loss / max(num_batches, 1)
                suppression_status = "ON" if config.use_dropout else "OFF"
                print(
                    f'  Epoch {epoch:3d}/{config.epochs}, Loss: {avg_loss:.4f}, Val NDCG: {val_ndcg:.4f} (Suppression: {suppression_status})')

            # Early stopping
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break

    # Load best model
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)

    return model, dataset.complete_edge_index, dataset.complete_edge_weight


def cross_validation_experiment_complete():
    args = parse_arguments()
    config = Config()
    config.update_from_args(args)
    config.setup_model_config()
    config.log_config()

    dataset = RecommendationDataset(name=config.dataset_name, data_path=f'dataset/{config.dataset_name}_raw')
    dataset.load_data().prepare_data()

    config.user_degrees, config.item_degrees = dataset.get_node_degrees()
    print("Preparing data with consistent encoding...")
    print(f"Processed data: {dataset.num_users} users, {dataset.num_items} items, {len(dataset.complete_df)} interactions")

    print(f"Train set: {len(dataset.train_val_df)} interactions")
    print(f"Test set: {len(dataset.test_df)} interactions")

    # 5-fold cross validation using temporal splits within training set
    print("\nStarting 5-fold cross validation with complete graph...")
    if config.use_dropout:
        print(f"Community edge suppression ENABLED - suppression strength: {config.community_suppression}")
        print(f"User dropout: {config.users_dec_perc_drop}, Item dropout: {config.items_dec_perc_drop}")
    else:
        print("Community edge suppression DISABLED")

    cv_results = []

    n_folds = 5
    for fold in range(n_folds):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        model = get_model(config=config, dataset=dataset)

        dataset.get_fold_data(fold)  # sets train_df and val_df for this fold

        model, complete_edge_index, complete_edge_weight = train_model(
            dataset=dataset,
            model=model,
            config=config,
            stage='loo',
        )

        # Evaluate on CV validation set
        if len(dataset.val_df) > 0:
            val_metrics = evaluate_model(
                model=model, dataset=dataset, k=10, stage='loo'
            )

            fold_results = {
                'fold': fold + 1,
                'val_ndcg': val_metrics['ndcg'],
                'val_recall': val_metrics['recall'],
                'val_precision': val_metrics['precision']
            }

            cv_results.append(fold_results)

            print(f"Fold {fold + 1} Results:")
            print(
                f"  Val - NDCG: {val_metrics['ndcg']:.4f}, Recall: {val_metrics['recall']:.4f}, Precision: {val_metrics['precision']:.4f}")

    # Calculate average CV results
    if cv_results:
        cv_df = pd.DataFrame(cv_results)
        cv_summary = {
            'val_ndcg_mean': cv_df['val_ndcg'].mean(),
            'val_ndcg_std': cv_df['val_ndcg'].std(),
            'val_recall_mean': cv_df['val_recall'].mean(),
            'val_recall_std': cv_df['val_recall'].std(),
            'val_precision_mean': cv_df['val_precision'].mean(),
            'val_precision_std': cv_df['val_precision'].std(),
        }

        print("\n" + "=" * 60)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Validation Set (5-fold CV):")
        print(f"  NDCG:      {cv_summary['val_ndcg_mean']:.4f} ± {cv_summary['val_ndcg_std']:.4f}")
        print(f"  Recall:    {cv_summary['val_recall_mean']:.4f} ± {cv_summary['val_recall_std']:.4f}")
        print(f"  Precision: {cv_summary['val_precision_mean']:.4f} ± {cv_summary['val_precision_std']:.4f}")
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
        model=final_model, dataset=dataset, k=10, stage='full_train'
    )

    print(f"\nFinal Test Set Results:")
    print(f"  NDCG:      {test_metrics['ndcg']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")

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
    parser.add_argument("--dataset_name", type=str, default='ml-100k')
    parser.add_argument("--users_top_percent", type=float, default=0.05)
    parser.add_argument("--items_top_percent", type=float, default=0.05)
    parser.add_argument("--users_dec_perc_drop", type=float, default=0.05)
    parser.add_argument("--items_dec_perc_drop", type=float, default=0.05)
    parser.add_argument("--community_suppression", type=float, default=0.6)
    parser.add_argument("--drop_only_power_nodes", type=bool, default=True)
    parser.add_argument("--use_dropout", type=bool, default=True)
    parser.add_argument("--k_th_fold", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    set_seed(42)  # For reproducibility
    cv_results, cv_summary, test_metrics, final_model = cross_validation_experiment_complete()

    print("\n" + "=" * 60)
    print("DETAILED FOLD RESULTS")
    print("=" * 60)
    for result in cv_results:
        print(f"Fold {result['fold']}:")
        print(
            f"  Val: NDCG={result['val_ndcg']:.4f}, Recall={result['val_recall']:.4f}, Precision={result['val_precision']:.4f}")
        print()

