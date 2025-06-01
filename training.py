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
from evaluation import evaluate_model_with_complete_graph
from models import LightGCN
import torch.optim as optim
from dataset import RecommendationDataset
from argparse import ArgumentParser

warnings.filterwarnings('ignore')


# Data preprocessing functions
def load_movielens_data(data_path='ml-100k'):
    """
    Load MovieLens 100K dataset
    Download from: https://grouplens.org/datasets/movielens/100k/
    """
    ratings_file = os.path.join(data_path, 'u.data')

    if not os.path.exists(ratings_file):
        print(f"MovieLens data not found at {data_path}")
        print("Please download MovieLens 100K dataset and extract to 'ml-100k' folder")

    # Load ratings data (user_id, item_id, rating, timestamp)
    ratings = pd.read_csv(ratings_file, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    return ratings


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
        )
    elif config.model_name == 'ItemKNN':
        return ItemKNN(
            num_items=dataset.num_items,
            topk=config.item_knn_topk,
            shrink=config.shrink,
        )
    elif config.model_name == 'MultiVAE':
        return MultiVAE(
            num_items=dataset.num_items,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def train_model_with_complete_graph(dataset, model, config, epochs=50, verbose=True):
    """
    Train LightGCN using the COMPLETE graph structure (train+val+test)
    but only compute loss on training interactions
    """

    complete_df = pd.concat([dataset.train_df, dataset.val_df, dataset.test_df], ignore_index=True)

    print(f"Complete graph: {len(complete_df)} interactions")
    print(f"Training interactions: {len(dataset.train_df)} (used for loss)")
    print(f"Val interactions: {len(dataset.val_df)} (used for evaluation)")
    print(f"Test interactions: {len(dataset.test_df)} (used for evaluation)")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # max ndcg
        factor=0.5,
        patience=10,
        min_lr=1e-5,
    )

    # Move data to device
    dataset.complete_edge_index.to(device)
    dataset.complete_edge_weight.to(device)

    # Create user positive items mapping for better negative sampling (from ALL interactions)
    user_positive_items = defaultdict(set)
    for _, row in complete_df.iterrows():
        user_positive_items[row['user_encoded']].add(row['item_encoded'])

    # Training parameters
    batch_size = config.batch_size
    best_val_ndcg = 0
    patience = config.patience
    patience_counter = 0

    model.train()

    for epoch in range(config.epochs):
        total_loss = 0
        num_batches = 0

        # Sample training batches from TRAINING data only (for loss computation)
        train_interactions = dataset.train_df.sample(n=min(len(dataset.train_df), batch_size * 10), replace=True)

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

            # Forward pass - use COMPLETE graph for embeddings
            user_emb, item_emb = model(dataset.complete_edge_index, dataset.complete_edge_weight)

            # Get embeddings for batch
            batch_user_emb = user_emb[batch_users]
            batch_pos_item_emb = item_emb[batch_pos_items]
            batch_neg_item_emb = item_emb[batch_neg_items]

            # Compute loss ONLY on training interactions
            loss = bpr_loss(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)

            # Add L2 regularization
            l2_reg = 0.01 * (
                    batch_user_emb.norm(2).pow(2) +
                    batch_pos_item_emb.norm(2).pow(2) +
                    batch_neg_item_emb.norm(2).pow(2)
            ) / batch_size

            total_loss_batch = loss + l2_reg

            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            scheduler.step(total_loss_batch.item())

            total_loss += total_loss_batch.item()
            num_batches += 1

        # Validation every 5 epochs
        if epoch % 5 == 0:
            val_metrics = evaluate_model_with_complete_graph(
                model, dataset.complete_edge_index, dataset.complete_edge_weight,
                dataset.val_df, dataset.train_df, dataset.num_users, dataset.num_items
            )
            val_ndcg = val_metrics['ndcg']

            if verbose and epoch % 10 == 0:
                avg_loss = total_loss / max(num_batches, 1)
                print(f'  Epoch {epoch:3d}/{epochs}, Loss: {avg_loss:.4f}, Val NDCG: {val_ndcg:.4f}')

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
    """Run cross validation with complete graph approach"""

    args = parse_arguments()
    config = Config()
    config.update_from_args(args)
    config.setup_model_config()
    config.log_config()

    dataset = RecommendationDataset(name=config.dataset_name, data_path=f'dataset/{config.dataset_name}_raw')
    dataset.load_data().prepare_data()

    print("Preparing data with consistent encoding...")
    print(f"Processed data: {dataset.num_users} users, {dataset.num_items} items, {len(dataset.complete_df)} interactions")

    # Split by interactions per user (not by random split)
    print("Splitting interactions by user...")

    print(f"Train set: {len(dataset.train_df)} interactions")
    print(f"Val set: {len(dataset.val_df)} interactions")
    print(f"Test set: {len(dataset.test_df)} interactions")

    # 5-fold cross validation using temporal splits within training set
    print("\nStarting 5-fold cross validation with complete graph...")

    # For CV, we need to do temporal splits on the training data
    cv_results = []

    # Group training data by user for CV splits
    user_train_interactions = {}
    for user_id, group in dataset.train_df.groupby('user_encoded'):
        user_train_interactions[user_id] = group.sort_values('timestamp').reset_index(drop=True)

    model = get_model(config=config, dataset=dataset)

    n_folds = 5
    for fold in range(n_folds):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # Create CV splits: use last 20% of each user's training data as CV validation
        cv_train_list = []
        cv_val_list = []

        for user_id, user_data in user_train_interactions.items():
            n_interactions = len(user_data)
            if n_interactions >= 2:
                # Use 80% for CV training, 20% for CV validation
                split_point = max(1, int(n_interactions * 0.8))
                cv_train_list.append(user_data.iloc[:split_point])
                cv_val_list.append(user_data.iloc[split_point:])
            else:
                cv_train_list.append(user_data)

        cv_train_df = pd.concat(cv_train_list, ignore_index=True)
        cv_val_df = pd.concat(cv_val_list, ignore_index=True) if cv_val_list else pd.DataFrame()

        print(f"Fold {fold + 1}: CV Train={len(cv_train_df)}, CV Val={len(cv_val_df)}")

        model, complete_edge_index, complete_edge_weight = train_model_with_complete_graph(
            dataset=dataset,
            model=model,
            config=config,
            epochs=50
        )

        # Evaluate on CV validation set
        if len(cv_val_df) > 0:
            val_metrics = evaluate_model_with_complete_graph(
                model, complete_edge_index, complete_edge_weight,
                cv_val_df, cv_train_df, dataset.num_users, dataset.num_items
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

    # Final training: use ALL data in graph, but only train+val for loss
    final_model, final_edge_index, final_edge_weight = train_model_with_complete_graph(
        dataset.train_df, dataset.val_df, dataset.test_df, dataset.num_users, dataset.num_items, epochs=100, verbose=True
    )

    # Evaluate on test set (excluding train+val interactions)
    train_val_df = pd.concat([dataset.train_df, dataset.val_df], ignore_index=True)
    test_metrics = evaluate_model_with_complete_graph(
        final_model, final_edge_index, final_edge_weight,
        dataset.test_df, train_val_df, dataset.num_users, dataset.num_items
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
    # Run complete graph cross-validation experiment
    cv_results, cv_summary, test_metrics, final_model = cross_validation_experiment_complete()

    # Print detailed results
    print("\n" + "=" * 60)
    print("DETAILED FOLD RESULTS")
    print("=" * 60)
    for result in cv_results:
        print(f"Fold {result['fold']}:")
        print(
            f"  Val: NDCG={result['val_ndcg']:.4f}, Recall={result['val_recall']:.4f}, Precision={result['val_precision']:.4f}")
        print()

