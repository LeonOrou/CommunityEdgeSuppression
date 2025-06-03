import numpy as np
import torch
from collections import defaultdict


def evaluate_model(model, dataset, stage='loo', k_values=[10, 20, 50, 100]):
    """
    Evaluate model using the complete graph structure but excluding training interactions
    stage: 'full_train' for full training evaluation, 'loo' for leave-one-out evaluation
    k_values: list of k_values values to evaluate (e.g., [10, 20, 50])
    """
    model.eval()

    # Convert k_values to list if it's a single value
    if isinstance(k_values, int):
        k_values = [k_values]

    # Sort k_values values to ensure we process from smallest to largest
    k_values = sorted(k_values)
    max_k = max(k_values)

    if stage == 'full_train':
        dataset.val_df = dataset.test_df

    # Create a set of training interactions for each user to exclude from evaluation
    train_user_items = defaultdict(set)
    for _, row in dataset.train_df.iterrows():
        train_user_items[row['user_encoded']].add(row['item_encoded'])

    # Initialize metric storage for each k_values
    metrics_by_k = {k_val: {
        'ndcg_scores': [],
        'recall_scores': [],
        'precision_scores': [],
        'mrr_scores': [],
        'hit_rate_scores': [],
        'all_recommended_items': set()
    } for k_val in k_values}

    # Store recommendation frequencies for Gini calculation
    item_recommendation_freq_by_k = {k_val: defaultdict(int) for k_val in k_values}

    with torch.no_grad():
        # Use the COMPLETE graph (including test edges) for embeddings
        user_emb, item_emb = model(dataset.complete_edge_index, dataset.complete_edge_weight)

        # Group test interactions by user
        user_test_items = dataset.val_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

        for user_id, true_items in user_test_items.items():
            if user_id >= dataset.num_users:
                continue

            # Get user embedding and compute scores for all items
            user_embedding = user_emb[user_id:user_id + 1]
            scores = torch.matmul(user_embedding, item_emb.T).squeeze().cpu().numpy()

            # Exclude items that the user interacted with during training
            train_items = list(train_user_items[user_id])
            scores_filtered = scores.copy()
            scores_filtered[train_items] = float('-inf')

            # Get top-max_k predictions (excluding training items)
            # We get the maximum k_values to avoid recomputing rankings
            top_max_k_items = np.argsort(scores_filtered)[::-1][:max_k]

            # Get full ranking for MRR calculation
            full_ranking = np.argsort(scores_filtered)[::-1]

            # Calculate metrics for each k_values value
            for k_val in k_values:
                # Get top-k_values items for this k_values value
                top_k_items = top_max_k_items[:k_val]

                # Add to recommended items set for coverage calculation
                metrics_by_k[k_val]['all_recommended_items'].update(top_k_items)

                # Update recommendation frequency
                for item in top_k_items:
                    item_recommendation_freq_by_k[k_val][item] += 1

                # Calculate metrics
                ndcg = calculate_ndcg(true_items, scores_filtered, k_val)
                recall = calculate_recall(true_items, top_k_items, k_val)
                precision = calculate_precision(true_items, top_k_items, k_val)
                mrr = calculate_mrr(true_items, full_ranking)  # MRR doesn't depend on k_values
                hit_rate = calculate_hit_rate(true_items, top_k_items, k_val)

                metrics_by_k[k_val]['ndcg_scores'].append(ndcg)
                metrics_by_k[k_val]['recall_scores'].append(recall)
                metrics_by_k[k_val]['precision_scores'].append(precision)
                metrics_by_k[k_val]['mrr_scores'].append(mrr)
                metrics_by_k[k_val]['hit_rate_scores'].append(hit_rate)

    # Compute final metrics for each k_values
    results = {}
    for k_val in k_values:
        # Calculate averages
        ndcg_scores = metrics_by_k[k_val]['ndcg_scores']
        recall_scores = metrics_by_k[k_val]['recall_scores']
        precision_scores = metrics_by_k[k_val]['precision_scores']
        mrr_scores = metrics_by_k[k_val]['mrr_scores']
        hit_rate_scores = metrics_by_k[k_val]['hit_rate_scores']

        # Calculate item coverage
        all_recommended_items = metrics_by_k[k_val]['all_recommended_items']
        item_coverage = calculate_item_coverage(list(all_recommended_items), dataset.num_items)

        # Calculate Gini index
        recommendation_counts = np.array(list(item_recommendation_freq_by_k[k_val].values()))
        if len(recommendation_counts) > 0:
            recommendation_probs = recommendation_counts / recommendation_counts.sum()
            gini_index = calculate_gini_coefficient(recommendation_probs)
        else:
            gini_index = 0.0

        # Simpson index as diversity measure
        simpson_index = 1.0 - gini_index

        results[k_val] = {
            'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'recall': np.mean(recall_scores) if recall_scores else 0.0,
            'precision': np.mean(precision_scores) if precision_scores else 0.0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'hit_rate': np.mean(hit_rate_scores) if hit_rate_scores else 0.0,
            'item_coverage': item_coverage,
            'gini_index': gini_index,
            'simpson_index': simpson_index
        }

    return results


# Evaluation metrics
def calculate_ndcg(y_true, y_scores, k=10):
    """Calculate NDCG@k_values"""
    if len(y_true) == 0:
        return 0.0

    # Create relevance scores (1 for relevant, 0 for non-relevant)
    relevance_scores = np.zeros(len(y_scores))
    relevance_scores[y_true] = 1

    # Get top-k_values predictions
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    top_k_relevance = relevance_scores[top_k_indices]

    if np.sum(top_k_relevance) == 0:
        return 0.0

    # Calculate DCG@k_values
    dcg = np.sum((2 ** top_k_relevance - 1) / np.log2(np.arange(2, k + 2)))

    # Calculate IDCG@k_values
    ideal_relevance = np.sort(relevance_scores)[::-1][:k]
    idcg = np.sum((2 ** ideal_relevance - 1) / np.log2(np.arange(2, k + 2)))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_recall(y_true, y_pred, k=10):
    """Calculate Recall@k_values"""
    if len(y_true) == 0:
        return 0.0

    top_k = set(y_pred[:k])
    relevant = set(y_true)

    if len(relevant) == 0:
        return 0.0

    return len(top_k.intersection(relevant)) / len(relevant)


def calculate_precision(y_true, y_pred, k=10):
    """Calculate Precision@k_values"""
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
    """Calculate Hit Rate@k_values"""
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


def calculate_gini_coefficient(probabilities):
    """
    Calculate Gini coefficient for recommendation distribution
    Higher values indicate more inequality (few items get most recommendations)
    Lower values indicate more equality (recommendations spread evenly)
    """
    if len(probabilities) == 0:
        return 0.0

    # Sort probabilities
    sorted_probs = np.sort(probabilities)
    n = len(sorted_probs)

    # Calculate Gini coefficient
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_probs)) / (n * np.sum(sorted_probs)) - (n + 1) / n

    return gini