import numpy as np
import torch
import scipy.sparse as sp
from collections import defaultdict

def evaluate_model(model, valid_data, k_list=[10, 20, 50], user_item_history=None, item_popularity=None, train_matrix=None):
    """
    Evaluate a link prediction model using various metrics.
    Parameters:
      - model: the model that makes link predictions.
      - test_data: dict of test user-item interactions.
      - k_list: list of top-k values to evaluate.
      - user_item_history: dict of known user items to filter out.
      - item_popularity: dict mapping item IDs to popularity scores.
      - train_matrix: sparse matrix of training data (for models that need it)
    Returns:
      - A dict with metrics for each k.
    """
    metrics = { k: {'ndcg': 0, 'recall': 0, 'hit': 0, 'avg_popularity': 0, 'gini': 0, 'coverage': 0} for k in k_list }
    item_recommended_count = defaultdict(int)
    total_items = 0  # will be set later
    user_count = 0

    for user_id, test_items in valid_data.items():
        user_count += 1
        if not test_items:
            continue

        with torch.no_grad():
            if hasattr(model, 'predict'):
                if train_matrix is not None and model.__class__.__name__ in ['ItemKNN', 'MultiVAE']:
                    predictions = model.predict([user_id], train_matrix)
                else:
                    predictions = model.predict(user_id)
            else:
                predictions = model(user_id)
        
        # Convert predictions to numpy array if it's a torch tensor
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
            if len(predictions.shape) > 1 and predictions.shape[0] == 1:
                predictions = predictions[0]  # Handle batch dimension of 1
        
        # Handle sparse predictions
        if sp.issparse(predictions):
            if hasattr(predictions, 'toarray'):
                predictions_array = predictions.toarray()
                if len(predictions_array.shape) > 1 and predictions_array.shape[0] == 1:
                    predictions_array = predictions_array[0]
                predictions = predictions_array
            else:
                # For other sparse formats, convert appropriately
                predictions = predictions.todense().A1

        # Set total_items if not already set
        if total_items == 0:
            total_items = len(predictions)

        # Filter out items from training data if history is provided
        if user_item_history and user_id in user_item_history:
            for item_id in user_item_history[user_id]:
                if item_id < len(predictions):
                    predictions[item_id] = float('-inf')

        for k in k_list:
            # Get top k item indices
            top_k_items = np.argsort(-predictions)[:k]
            
            # Count recommended items for coverage and Gini calculation
            for item_id in top_k_items:
                item_recommended_count[item_id] += 1
            
            # Set of relevant items (test items)
            relevant_items = set(test_items)
            
            # Calculate Hit rate (1 if at least one hit, 0 otherwise)
            hit = int(any(item in relevant_items for item in top_k_items))
            metrics[k]['hit'] += hit
            
            # Calculate Recall
            recall = len(set(top_k_items) & relevant_items) / len(relevant_items) if relevant_items else 0
            metrics[k]['recall'] += recall
            
            # Calculate NDCG
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
            dcg = 0
            for i, item in enumerate(top_k_items):
                if item in relevant_items:
                    dcg += 1.0 / np.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[k]['ndcg'] += ndcg
            
            # Calculate average popularity if popularity data is provided
            if item_popularity:
                avg_pop = sum(item_popularity.get(item, 0) for item in top_k_items) / k
                metrics[k]['avg_popularity'] += avg_pop

    # Average metrics over all users
    for k in k_list:
        metrics[k]['ndcg'] /= user_count
        metrics[k]['recall'] /= user_count
        metrics[k]['hit'] /= user_count
        if item_popularity:
            metrics[k]['avg_popularity'] /= user_count
        
        # Calculate Coverage
        metrics[k]['coverage'] = len(item_recommended_count) / total_items if total_items > 0 else 0
        
        # Calculate Gini coefficient
        if total_items > 0:
            # Get recommendation counts (0 for items never recommended)
            counts = np.zeros(total_items)
            for item_id, count in item_recommended_count.items():
                if item_id < total_items:
                    counts[item_id] = count
            
            # Sort counts
            counts = np.sort(counts)
            indices = np.arange(1, total_items + 1)
            gini = 1 - (2 * np.sum((total_items + 1 - indices) * counts)) / (total_items * np.sum(counts))
            metrics[k]['gini'] = gini
    
    # Flatten metrics for simplicity if only one k value
    if len(k_list) == 1:
        metrics = metrics[k_list[0]]
        
    return metrics

def calculate_ndcg(ranked_items, relevant_items, k):
    """
    Calculate NDCG given a ranked list of items and relevant items.
    """
    dcg = 0
    for i, item in enumerate(ranked_items[:k]):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
    return dcg / idcg if idcg > 0 else 0

def calculate_recall(predicted_items, relevant_items):
    """
    Calculate Recall.
    """
    return len(set(predicted_items) & relevant_items) / len(relevant_items) if relevant_items else 0

def calculate_hit_rate(predicted_items, relevant_items):
    """
    Calculate Hit Rate (1 if at least one hit, else 0).
    """
    return int(any(item in relevant_items for item in predicted_items))


def precalculate_average_popularity(dataset):
    # calculate how many links each item has
    # count occurences of items with torch.bincount
    # return a dict with item_id as key and popularity as value
    item_popularity = torch.bincount(dataset.inter_feat['item_id'].values)
    item_popularity = item_popularity.cpu().numpy()
    item_popularity_dict = {i: item_popularity[i] for i in range(len(item_popularity))}
    return item_popularity_dict


