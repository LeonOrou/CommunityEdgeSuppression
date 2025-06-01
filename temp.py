import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
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

warnings.filterwarnings('ignore')


# Custom LightGCN implementation with edge weights
class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__(aggr='add')
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, edge_index, edge_weight=None):
        # Get initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        # Combine user and item embeddings
        x = torch.cat([user_emb, item_emb], dim=0)

        # Store embeddings from each layer
        embeddings = [x]

        for _ in range(self.num_layers):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            embeddings.append(x)

        # Average embeddings across all layers
        final_embedding = torch.stack(embeddings, dim=0).mean(dim=0)

        # Split back into user and item embeddings
        user_final = final_embedding[:self.num_users]
        item_final = final_embedding[self.num_users:]

        return user_final, item_final

    def message(self, x_j, edge_weight):
        # Apply edge weights to messages
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out, x):
        return aggr_out


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
        # Create synthetic data for demonstration
        return create_synthetic_data()

    # Load ratings data (user_id, item_id, rating, timestamp)
    ratings = pd.read_csv(ratings_file, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    return ratings


def create_synthetic_data():
    """Create synthetic MovieLens-like data for demonstration"""
    np.random.seed(42)

    num_users = 1000
    num_items = 500
    num_interactions = 50000

    users = np.random.randint(1, num_users + 1, num_interactions)
    items = np.random.randint(1, num_items + 1, num_interactions)
    ratings = np.random.choice([1, 2, 3, 4, 5], num_interactions,
                               p=[0.05, 0.1, 0.2, 0.35, 0.3])  # Bias toward higher ratings

    df = pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'rating': ratings,
        'timestamp': np.random.randint(1000000000, 1500000000, num_interactions)
    })

    # Remove duplicates, keeping the latest rating
    df = df.sort_values('timestamp').drop_duplicates(['user_id', 'item_id'], keep='last')

    return df


def prepare_data(ratings_df, min_interactions=5):
    """Prepare data for training"""
    # Filter users and items with minimum interactions
    # shuffle the DataFrame
    ratings_df = ratings_df.sample(frac=1, random_state=42).reset_index(drop=True)

    user_counts = ratings_df['user_id'].value_counts()
    item_counts = ratings_df['item_id'].value_counts()

    valid_users = user_counts[user_counts >= min_interactions].index
    valid_items = item_counts[item_counts >= min_interactions].index

    filtered_df = ratings_df[
        (ratings_df['user_id'].isin(valid_users)) &
        (ratings_df['item_id'].isin(valid_items))
        ].copy()

    # Encode user and item IDs
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


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """Bayesian Personalized Ranking loss"""
    pos_scores = (user_emb * pos_item_emb).sum(dim=1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=1)
    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
    return loss


def sample_negative_items(user_ids, pos_item_ids, num_items, num_negatives=1):
    """Sample negative items for each user-positive item pair"""
    neg_items = []

    for user_id, pos_item_id in zip(user_ids, pos_item_ids):
        user_neg_items = []
        for _ in range(num_negatives):
            neg_item = np.random.randint(0, num_items)
            # Simple negative sampling (could be improved with more sophisticated methods)
            user_neg_items.append(neg_item)
        neg_items.extend(user_neg_items)

    return torch.tensor(neg_items, dtype=torch.long)


# Evaluation metrics
def calculate_ndcg(y_true, y_scores, k=10):
    """Calculate NDCG@k"""
    if len(y_true) == 0:
        return 0.0

    # Create relevance scores (1 for relevant, 0 for non-relevant)
    relevance_scores = np.zeros(len(y_scores))
    relevance_scores[y_true] = 1

    # Get top-k predictions
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    top_k_relevance = relevance_scores[top_k_indices]

    if np.sum(top_k_relevance) == 0:
        return 0.0

    # Calculate DCG@k
    dcg = np.sum((2 ** top_k_relevance - 1) / np.log2(np.arange(2, k + 2)))

    # Calculate IDCG@k
    ideal_relevance = np.sort(relevance_scores)[::-1][:k]
    idcg = np.sum((2 ** ideal_relevance - 1) / np.log2(np.arange(2, k + 2)))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_recall(y_true, y_pred, k=10):
    """Calculate Recall@k"""
    if len(y_true) == 0:
        return 0.0

    top_k = set(y_pred[:k])
    relevant = set(y_true)

    if len(relevant) == 0:
        return 0.0

    return len(top_k.intersection(relevant)) / len(relevant)


def calculate_precision(y_true, y_pred, k=10):
    """Calculate Precision@k"""
    if len(y_true) == 0:
        return 0.0

    top_k = set(y_pred[:k])
    relevant = set(y_true)

    if len(top_k) == 0:
        return 0.0

    return len(top_k.intersection(relevant)) / len(top_k)


def evaluate_model(model, edge_index, edge_weight, test_df, num_users, num_items, k=10):
    """Evaluate model on test set"""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        user_emb, item_emb = model(edge_index, edge_weight)

        # Group test interactions by user
        user_interactions = test_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

        ndcg_scores = []
        recall_scores = []
        precision_scores = []

        for user_id, true_items in user_interactions.items():
            if user_id >= num_users:
                continue

            # Get user embedding and compute scores
            user_embedding = user_emb[user_id:user_id + 1]
            scores = torch.matmul(user_embedding, item_emb.T).squeeze().cpu().numpy()

            # Get top-k predictions
            top_k_items = np.argsort(scores)[::-1][:k]

            # Calculate metrics
            ndcg = calculate_ndcg(true_items, scores, k)
            recall = calculate_recall(true_items, top_k_items, k)
            precision = calculate_precision(true_items, top_k_items, k)

            ndcg_scores.append(ndcg)
            recall_scores.append(recall)
            precision_scores.append(precision)

    return {
        'ndcg': np.mean(ndcg_scores),
        'recall': np.mean(recall_scores),
        'precision': np.mean(precision_scores)
    }


def split_dataset(df, test_size=0.2, random_state=42):
    test_split_idx = int(len(df) * (1 - test_size))
    train_val_df = df[:test_split_idx].copy()
    test_df = df[test_split_idx:].copy()

    return train_val_df, test_df


# Training loop
def train_lightgcn_fold(train_df, val_df, num_users, num_items, epochs=50, verbose=True):
    """Train LightGCN for one fold"""

    # Create graph data
    edge_index, edge_weight = create_bipartite_graph(train_df, num_users, num_items)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=64,
        num_layers=3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Move data to device
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # Training parameters
    batch_size = 1024
    best_val_ndcg = 0
    patience = 10
    patience_counter = 0

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # Sample training batches
        train_interactions = train_df.sample(n=min(len(train_df), batch_size * 10))

        for i in range(0, len(train_interactions), batch_size):
            batch = train_interactions.iloc[i:i + batch_size]

            if len(batch) == 0:
                continue

            # Get batch data
            batch_users = torch.tensor(batch['user_encoded'].values, dtype=torch.long).to(device)
            batch_pos_items = torch.tensor(batch['item_encoded'].values, dtype=torch.long).to(device)
            batch_neg_items = sample_negative_items(
                batch['user_encoded'].values,
                batch['item_encoded'].values,
                num_items
            ).to(device)

            # Forward pass
            user_emb, item_emb = model(edge_index, edge_weight)

            # Get embeddings for batch
            batch_user_emb = user_emb[batch_users]
            batch_pos_item_emb = item_emb[batch_pos_items]
            batch_neg_item_emb = item_emb[batch_neg_items]

            # Compute loss
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

            total_loss += total_loss_batch.item()
            num_batches += 1

        # Validation every 5 epochs
        if epoch % 5 == 0:
            val_metrics = evaluate_model(model, edge_index, edge_weight, val_df, num_users, num_items)
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

    return model, edge_index, edge_weight


def cross_validation_experiment():
    """Run 5-fold cross validation experiment"""
    print("Loading MovieLens data...")
    ratings_df = load_movielens_data()
    print(f"Loaded {len(ratings_df)} ratings")

    print("Preparing data...")
    processed_df, num_users, num_items, user_encoder, item_encoder = prepare_data(ratings_df)
    print(f"Processed data: {num_users} users, {num_items} items, {len(processed_df)} interactions")

    # Split dataset: 80% for CV, 20% for final test
    print("Splitting dataset...")
    train_val_df, test_df = split_dataset(processed_df, test_size=0.2)

    print(f"Train+Val set: {len(train_val_df)} interactions")
    print(f"Test set: {len(test_df)} interactions")

    # 5-fold cross validation on the 80% data
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    print("\nStarting 5-fold cross validation...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_df)):
        print(f"\n--- Fold {fold + 1}/5 ---")

        # Split fold data
        fold_train_df = train_val_df.iloc[train_idx].copy()
        fold_val_df = train_val_df.iloc[val_idx].copy()

        print(f"Fold {fold + 1}: Train={len(fold_train_df)}, Val={len(fold_val_df)}")

        # Train model for this fold
        model, edge_index, edge_weight = train_lightgcn_fold(
            fold_train_df, fold_val_df, num_users, num_items, epochs=50
        )

        # Evaluate on validation set
        val_metrics = evaluate_model(model, edge_index, edge_weight, fold_val_df, num_users, num_items)

        # Also evaluate on training set for comparison
        train_metrics = evaluate_model(model, edge_index, edge_weight, fold_train_df, num_users, num_items)

        fold_results = {
            'fold': fold + 1,
            'train_ndcg': train_metrics['ndcg'],
            'train_recall': train_metrics['recall'],
            'train_precision': train_metrics['precision'],
            'val_ndcg': val_metrics['ndcg'],
            'val_recall': val_metrics['recall'],
            'val_precision': val_metrics['precision']
        }

        cv_results.append(fold_results)

        print(f"Fold {fold + 1} Results:")
        print(
            f"  Train - NDCG: {train_metrics['ndcg']:.4f}, Recall: {train_metrics['recall']:.4f}, Precision: {train_metrics['precision']:.4f}")
        print(
            f"  Val   - NDCG: {val_metrics['ndcg']:.4f}, Recall: {val_metrics['recall']:.4f}, Precision: {val_metrics['precision']:.4f}")

    # Calculate average CV results
    cv_df = pd.DataFrame(cv_results)
    cv_summary = {
        'train_ndcg_mean': cv_df['train_ndcg'].mean(),
        'train_ndcg_std': cv_df['train_ndcg'].std(),
        'val_ndcg_mean': cv_df['val_ndcg'].mean(),
        'val_ndcg_std': cv_df['val_ndcg'].std(),
        'train_recall_mean': cv_df['train_recall'].mean(),
        'train_recall_std': cv_df['train_recall'].std(),
        'val_recall_mean': cv_df['val_recall'].mean(),
        'val_recall_std': cv_df['val_recall'].std(),
        'train_precision_mean': cv_df['train_precision'].mean(),
        'train_precision_std': cv_df['train_precision'].std(),
        'val_precision_mean': cv_df['val_precision'].mean(),
        'val_precision_std': cv_df['val_precision'].std(),
    }

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Training Set:")
    print(f"  NDCG:      {cv_summary['train_ndcg_mean']:.4f} ± {cv_summary['train_ndcg_std']:.4f}")
    print(f"  Recall:    {cv_summary['train_recall_mean']:.4f} ± {cv_summary['train_recall_std']:.4f}")
    print(f"  Precision: {cv_summary['train_precision_mean']:.4f} ± {cv_summary['train_precision_std']:.4f}")
    print(f"\nValidation Set:")
    print(f"  NDCG:      {cv_summary['val_ndcg_mean']:.4f} ± {cv_summary['val_ndcg_std']:.4f}")
    print(f"  Recall:    {cv_summary['val_recall_mean']:.4f} ± {cv_summary['val_recall_std']:.4f}")
    print(f"  Precision: {cv_summary['val_precision_mean']:.4f} ± {cv_summary['val_precision_std']:.4f}")

    # Train final model on all training data and evaluate on test set
    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("=" * 60)

    print("Training final model on all training data...")
    # Use a small validation set from training data for early stopping
    final_train_df, final_val_df = train_test_split(train_val_df, test_size=0.1, random_state=42)

    final_model, final_edge_index, final_edge_weight = train_lightgcn_fold(
        final_train_df, final_val_df, num_users, num_items, epochs=100
    )

    # Evaluate on test set
    test_metrics = evaluate_model(final_model, final_edge_index, final_edge_weight, test_df, num_users, num_items)

    print(f"\nFinal Test Set Results:")
    print(f"  NDCG:      {test_metrics['ndcg']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")

    return cv_results, cv_summary, test_metrics, final_model


# Function to get recommendations
def get_recommendations(model, user_id, user_emb, item_emb, k=10, exclude_seen=None):
    """Get top-k recommendations for a user"""
    model.eval()
    with torch.no_grad():
        user_embedding = user_emb[user_id:user_id + 1]
        scores = torch.matmul(user_embedding, item_emb.T).squeeze()

        if exclude_seen is not None:
            scores[exclude_seen] = float('-inf')

        top_items = torch.topk(scores, k=k).indices
        top_scores = torch.topk(scores, k=k).values

    return top_items.cpu().numpy(), top_scores.cpu().numpy()


if __name__ == "__main__":
    # Run cross-validation experiment
    cv_results, cv_summary, test_metrics, final_model = cross_validation_experiment()

    # Print detailed results
    print("\n" + "=" * 60)
    print("DETAILED FOLD RESULTS")
    print("=" * 60)
    for result in cv_results:
        print(f"Fold {result['fold']}:")
        print(
            f"  Train: NDCG={result['train_ndcg']:.4f}, Recall={result['train_recall']:.4f}, Precision={result['train_precision']:.4f}")
        print(
            f"  Val:   NDCG={result['val_ndcg']:.4f}, Recall={result['val_recall']:.4f}, Precision={result['val_precision']:.4f}")
        print()