import numpy as np
import torch
from collections import defaultdict


def evaluate_model_with_complete_graph(model, complete_edge_index, complete_edge_weight,
                                       test_df, train_df, num_users, num_items, k=10):
    """
    Evaluate model using the complete graph structure but excluding training interactions
    """
    model.eval()
    device = next(model.parameters()).device

    # Create a set of training interactions for each user to exclude from evaluation
    train_user_items = defaultdict(set)
    for _, row in train_df.iterrows():
        train_user_items[row['user_encoded']].add(row['item_encoded'])

    with torch.no_grad():
        # Use the COMPLETE graph (including test edges) for embeddings
        user_emb, item_emb = model(complete_edge_index, complete_edge_weight)

        # Group test interactions by user
        user_test_items = test_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

        ndcg_scores = []
        recall_scores = []
        precision_scores = []

        for user_id, true_items in user_test_items.items():
            if user_id >= num_users:
                continue

            # Get user embedding and compute scores for all items
            user_embedding = user_emb[user_id:user_id + 1]
            scores = torch.matmul(user_embedding, item_emb.T).squeeze().cpu().numpy()

            # Exclude items that the user interacted with during training
            train_items = list(train_user_items[user_id])
            scores_filtered = scores.copy()
            scores_filtered[train_items] = float('-inf')

            # Get top-k predictions (excluding training items)
            top_k_items = np.argsort(scores_filtered)[::-1][:k]

            # Calculate metrics
            ndcg = calculate_ndcg(true_items, scores_filtered, k)
            recall = calculate_recall(true_items, top_k_items, k)
            precision = calculate_precision(true_items, top_k_items, k)

            ndcg_scores.append(ndcg)
            recall_scores.append(recall)
            precision_scores.append(precision)

    return {
        'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'recall': np.mean(recall_scores) if recall_scores else 0.0,
        'precision': np.mean(precision_scores) if precision_scores else 0.0
    }


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


def calculate_mrr(y_true, y_pred):
    """Calculate Mean Reciprocal Rank (MRR)"""
    if len(y_true) == 0:
        return 0.0

    ranks = []
    for item in y_true:
        if item in y_pred:
            rank = np.where(y_pred == item)[0][0] + 1  # +1 for 1-based index
            ranks.append(1 / rank)

    if not ranks:
        return 0.0

    return np.mean(ranks)


def calculate_hit_rate(y_true, y_pred, k=10):
    """Calculate Hit Rate@k"""
    if len(y_true) == 0:
        return 0.0

    top_k = set(y_pred[:k])
    relevant = set(y_true)

    return 1.0 if top_k.intersection(relevant) else 0.0


def calculate_item_coverage(y_pred, num_items):
    """Calculate Item Coverage"""
    if len(y_pred) == 0:
        return 0.0

    unique_items = set(y_pred)
    return len(unique_items) / num_items


def calculate_gini_index(y_true, y_pred):
    """Calculate Gini Index"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0

    # Sort predictions by scores
    sorted_indices = np.argsort(y_pred)[::-1]
    sorted_y_true = np.array(y_true)[sorted_indices]

    n = len(sorted_y_true)
    cumulative_gain = np.cumsum(sorted_y_true)
    total_gain = cumulative_gain[-1]

    if total_gain == 0:
        return 0.0

    gini = (2 * np.sum(cumulative_gain) / total_gain - (n + 1)) / n
    return gini


def calculate_simpson_index(y_true, y_pred):
    """Calculate Simpson Index"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0

    top_k = set(y_pred)
    relevant = set(y_true)

    intersection = len(top_k.intersection(relevant))
    union = len(top_k.union(relevant))

    if union == 0:
        return 0.0

    return intersection / union



