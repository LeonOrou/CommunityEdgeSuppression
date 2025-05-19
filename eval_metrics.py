import numpy as np
import torch
import scipy.sparse as sp
from collections import defaultdict


def evaluate_model(model, test_loader, device, k_list=[10, 20, 50], user_item_history=None,
                   item_popularity=None, train_matrix=None):
    """
    GPU-optimized evaluation function that processes users in batches.
    """
    model.eval()
    metrics = {k: {'ndcg': 0, 'recall': 0, 'hit': 0, 'avg_popularity': 0, 'gini': 0, 'coverage': 0} for k in k_list}

    # Track item recommendations for coverage/gini on GPU
    n_items = getattr(model, 'n_items', train_matrix.shape[1] if train_matrix is not None else 0)
    item_recommended = torch.zeros(n_items, device=device)
    user_count = 0

    with torch.no_grad():
        for batch in test_loader:
            # Process each batch of test data
            users, items, ratings = batch
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)

            # Group test items by user efficiently
            user_ids, indices = torch.unique(users, return_inverse=True)
            user_count += len(user_ids)

            # Get predictions for all users in batch at once
            if hasattr(model, 'predict_batch'):
                predictions = model.predict_batch(user_ids)
            elif hasattr(model, 'predict'):
                if train_matrix is not None and hasattr(model, '__class__') and model.__class__.__name__ in ['ItemKNN',
                                                                                                             'MultiVAE']:
                    predictions = model.predict(user_ids.cpu().numpy(), train_matrix)
                    if isinstance(predictions, torch.Tensor):
                        predictions = predictions.to(device)
                    else:
                        predictions = torch.tensor(predictions, device=device)
                else:
                    # For models that predict one user at a time
                    predictions = []
                    for user_id in user_ids:
                        pred = model.predict(user_id.unsqueeze(0))
                        if isinstance(pred, torch.Tensor):
                            predictions.append(pred.squeeze())
                        else:
                            predictions.append(torch.tensor(pred, device=device))
                    predictions = torch.stack(predictions)
            else:
                # Default to embedding dot product
                user_embeddings, item_embeddings = model.forward()
                predictions = torch.matmul(user_embeddings[user_ids], item_embeddings.T)

            # Filter seen items efficiently with mask operations
            if user_item_history:
                for i, user_id in enumerate(user_ids.cpu().numpy()):
                    if user_id in user_item_history:
                        history_items = torch.tensor(user_item_history[user_id], device=device)
                        predictions[i].index_fill_(0, history_items, float('-inf'))

            # Process each user's metrics
            for i, user_id in enumerate(user_ids.cpu().numpy()):
                # Get relevant items for this user from the batch
                mask = (users == user_id) & (ratings > 0)
                relevant_items = items[mask]

                if len(relevant_items) == 0:
                    continue

                # Calculate metrics for each k
                for k in k_list:
                    # Get top-k items for this user
                    _, top_indices = torch.topk(predictions[i], k)

                    # Update recommendation counts atomically
                    item_recommended.index_add_(0, top_indices, torch.ones(len(top_indices), device=device))

                    # Calculate metrics on GPU
                    # Convert to sets for intersection operations
                    top_set = set(top_indices.cpu().tolist())
                    relevant_set = set(relevant_items.cpu().tolist())

                    # Hit rate
                    hit = 1.0 if len(top_set & relevant_set) > 0 else 0.0
                    metrics[k]['hit'] += hit

                    # Recall
                    recall = len(top_set & relevant_set) / len(relevant_set)
                    metrics[k]['recall'] += recall

                    # NDCG
                    idcg = sum(1.0 / torch.log2(torch.tensor(i + 2.0)) for i in range(min(len(relevant_set), k)))
                    dcg = sum(1.0 / torch.log2(torch.tensor(i + 2.0))
                              for i, item in enumerate(top_indices.cpu().tolist()) if item in relevant_set)
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    metrics[k]['ndcg'] += ndcg

                    # Popularity
                    if item_popularity:
                        avg_pop = sum(item_popularity.get(item.item(), 0) for item in top_indices) / k
                        metrics[k]['avg_popularity'] += avg_pop

    # Move recommendation counts to CPU for final calculations
    item_recommended = item_recommended.cpu()

    # Normalize metrics
    for k in k_list:
        for metric in ['ndcg', 'recall', 'hit', 'avg_popularity']:
            metrics[k][metric] /= user_count if user_count > 0 else 1

        # Coverage
        metrics[k]['coverage'] = torch.count_nonzero(item_recommended).item() / n_items if n_items > 0 else 0

        # Gini coefficient
        if n_items > 0:
            # Sort counts
            sorted_counts = torch.sort(item_recommended)[0]
            indices = torch.arange(1, n_items + 1, dtype=torch.float)
            gini = 1 - (2 * torch.sum((n_items + 1 - indices) * sorted_counts)) / (
                        n_items * torch.sum(sorted_counts) + 1e-8)
            metrics[k]['gini'] = gini.item()

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


