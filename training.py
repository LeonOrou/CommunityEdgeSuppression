import torch
from torch_geometric.utils import degree, add_self_loops
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
import os
from collections import defaultdict
import warnings

from evaluation import evaluate_model_with_complete_graph
from models import LightGCN
import torch.optim as optim

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


def prepare_data_with_consistent_encoding(ratings_df, min_interactions=5):
    """
    Prepare data ensuring ALL users and items are encoded consistently
    across train/validation/test splits
    """
    # Filter users and items with minimum interactions FIRST
    ratings_df = ratings_df.sample(frac=1, random_state=42).reset_index(drop=True)

    user_counts = ratings_df['user_id'].value_counts()
    item_counts = ratings_df['item_id'].value_counts()

    valid_users = user_counts[user_counts >= min_interactions].index
    valid_items = item_counts[item_counts >= min_interactions].index

    filtered_df = ratings_df[
        (ratings_df['user_id'].isin(valid_users)) &
        (ratings_df['item_id'].isin(valid_items))
        ].copy()

    # Encode ALL users and items that appear in the dataset
    # This ensures consistent encoding across all splits
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    filtered_df['user_encoded'] = user_encoder.fit_transform(filtered_df['user_id'])
    filtered_df['item_encoded'] = item_encoder.fit_transform(filtered_df['item_id'])

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    return filtered_df, num_users, num_items, user_encoder, item_encoder


def create_bipartite_graph(df, num_users, num_items, rating_weights=None):
    """Create bipartite graph with edge weights based on ratings"""

    # Default rating weights (can be customized)
    if rating_weights is None:
        rating_weights = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}

    users = df['user_encoded'].values
    items = df['item_encoded'].values + num_users  # Offset items by num_users
    ratings = df['rating'].values

    # Create edge weights based on ratings
    edge_weights = np.array([rating_weights.get(r, 0.5) for r in ratings])

    # Create bidirectional edges (user->item and item->user)
    edge_index = torch.tensor([
        np.concatenate([users, items]),
        np.concatenate([items, users])
    ], dtype=torch.long)

    edge_weight = torch.tensor(
        np.concatenate([edge_weights, edge_weights]),
        dtype=torch.float
    )

    return edge_index, edge_weight


def split_interactions_by_user(df, test_ratio=0.2, val_ratio=0.1):
    """
    Split interactions for each user into train/val/test
    This ensures all users appear in training data
    """
    train_list = []
    val_list = []
    test_list = []

    for user_id, group in df.groupby('user_encoded'):
        user_interactions = group.sort_values('timestamp').reset_index(drop=True)
        n_interactions = len(user_interactions)

        if n_interactions < 3:
            # If user has very few interactions, put most in training, 1 in test
            train_list.append(user_interactions.iloc[:-1])
            test_list.append(user_interactions.iloc[-1:])
            continue

        # Calculate split points
        n_test = max(1, int(n_interactions * test_ratio))
        n_val = max(1, int(n_interactions * val_ratio))
        n_train = n_interactions - n_test - n_val

        if n_train < 1:
            n_train = 1
            n_val = max(0, n_interactions - n_train - n_test)
            n_test = n_interactions - n_train - n_val

        # Split interactions (chronologically)
        train_interactions = user_interactions.iloc[:n_train]
        val_interactions = user_interactions.iloc[n_train:n_train + n_val]
        test_interactions = user_interactions.iloc[n_train + n_val:]

        train_list.append(train_interactions)
        if len(val_interactions) > 0:
            val_list.append(val_interactions)
        if len(test_interactions) > 0:
            test_list.append(test_interactions)

    train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

    return train_df, val_df, test_df


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


def train_lightgcn_with_complete_graph(train_df, val_df, test_df, num_users, num_items, epochs=50, verbose=True):
    """
    Train LightGCN using the COMPLETE graph structure (train+val+test)
    but only compute loss on training interactions
    """

    # CRITICAL: Create the COMPLETE graph including ALL interactions
    complete_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    complete_edge_index, complete_edge_weight = create_bipartite_graph(complete_df, num_users, num_items)

    print(f"Complete graph: {len(complete_df)} interactions")
    print(f"Training interactions: {len(train_df)} (used for loss)")
    print(f"Val interactions: {len(val_df)} (used for evaluation)")
    print(f"Test interactions: {len(test_df)} (used for evaluation)")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=64,
        num_layers=3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # max ndcg
        factor=0.5,
        patience=10,
        min_lr=1e-5,
    )

    # Move data to device
    complete_edge_index = complete_edge_index.to(device)
    complete_edge_weight = complete_edge_weight.to(device)

    # Create user positive items mapping for better negative sampling (from ALL interactions)
    user_positive_items = defaultdict(set)
    for _, row in complete_df.iterrows():
        user_positive_items[row['user_encoded']].add(row['item_encoded'])

    # Training parameters
    batch_size = 1024
    best_val_ndcg = 0
    patience = 15
    patience_counter = 0

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # Sample training batches from TRAINING data only (for loss computation)
        train_interactions = train_df.sample(n=min(len(train_df), batch_size * 10), replace=True)

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
                num_items,
                user_positive_items=user_positive_items
            ).to(device)

            # Forward pass - use COMPLETE graph for embeddings
            user_emb, item_emb = model(complete_edge_index, complete_edge_weight)

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
                model, complete_edge_index, complete_edge_weight,
                val_df, train_df, num_users, num_items
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

    return model, complete_edge_index, complete_edge_weight


def cross_validation_experiment_complete():
    """Run cross validation with complete graph approach"""
    print("Loading MovieLens data...")
    ratings_df = load_movielens_data()
    print(f"Loaded {len(ratings_df)} ratings")

    print("Preparing data with consistent encoding...")
    processed_df, num_users, num_items, user_encoder, item_encoder = prepare_data_with_consistent_encoding(ratings_df)
    print(f"Processed data: {num_users} users, {num_items} items, {len(processed_df)} interactions")

    # Split by interactions per user (not by random split)
    print("Splitting interactions by user...")
    train_df, val_df, test_df = split_interactions_by_user(processed_df, test_ratio=0.2, val_ratio=0.1)

    print(f"Train set: {len(train_df)} interactions")
    print(f"Val set: {len(val_df)} interactions")
    print(f"Test set: {len(test_df)} interactions")

    # 5-fold cross validation using temporal splits within training set
    print("\nStarting 5-fold cross validation with complete graph...")

    # For CV, we need to do temporal splits on the training data
    cv_results = []

    # Group training data by user for CV splits
    user_train_interactions = {}
    for user_id, group in train_df.groupby('user_encoded'):
        user_train_interactions[user_id] = group.sort_values('timestamp').reset_index(drop=True)

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

        # Train model with complete graph (CV train + CV val + original val + test)
        model, complete_edge_index, complete_edge_weight = train_lightgcn_with_complete_graph(
            cv_train_df, cv_val_df, pd.concat([val_df, test_df]), num_users, num_items, epochs=50
        )

        # Evaluate on CV validation set
        if len(cv_val_df) > 0:
            val_metrics = evaluate_model_with_complete_graph(
                model, complete_edge_index, complete_edge_weight,
                cv_val_df, cv_train_df, num_users, num_items
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
    final_model, final_edge_index, final_edge_weight = train_lightgcn_with_complete_graph(
        train_df, val_df, test_df, num_users, num_items, epochs=100, verbose=True
    )

    # Evaluate on test set (excluding train+val interactions)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    test_metrics = evaluate_model_with_complete_graph(
        final_model, final_edge_index, final_edge_weight,
        test_df, train_val_df, num_users, num_items
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

