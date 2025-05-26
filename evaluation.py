import numpy as np
import torch
import scipy.sparse as sp
from collections import defaultdict
import math


def evaluate_model(model, test_loader, device, k_list=[10, 20, 50, 100], user_item_history=None,
                   item_popularity=None, train_matrix=None):
    """
    GPU-optimized evaluation function that processes users in batches,
    computing metrics vectorized over users.
    """
    model.eval()
    metrics = {k: {'ndcg': 0, 'recall': 0, 'hit': 0, 'avg_popularity': 0, 'gini': 0, 'coverage': 0} for k in k_list}
    n_items = getattr(model, 'n_items', train_matrix.shape[1] if train_matrix is not None else 0)
    item_recommended = torch.zeros(n_items, device=device)
    user_total = 0

    with torch.no_grad():
        for batch in test_loader:
            users, items, ratings = batch
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)
            
            # Get unique users in batch
            user_ids, inv_idx = torch.unique(users, return_inverse=True)
            batch_size = len(user_ids)
            user_total += batch_size
            
            # Get predictions for all users in batch at once
            if hasattr(model, 'predict_batch'):
                predictions = model.predict_batch(user_ids)
            elif hasattr(model, 'predict'):
                if train_matrix is not None and hasattr(model, '__class__') and model.__class__.__name__ in ['ItemKNN', 'MultiVAE', 'LightGCN']:
                    predictions = model.predict(user_ids, train_matrix)
                    predictions = predictions.to(device) if isinstance(predictions, torch.Tensor) else torch.tensor(predictions, device=device)
                else:
                    preds = []
                    for user_id in user_ids:
                        pred = model.predict(user_id.unsqueeze(0))
                        preds.append(pred.squeeze() if isinstance(pred, torch.Tensor) else torch.tensor(pred, device=device))
                    predictions = torch.stack(preds)
            else:
                user_embeddings, item_embeddings = model.forward()
                predictions = torch.matmul(user_embeddings[user_ids], item_embeddings.T)
            
            # Filter seen items
            if user_item_history:
                for i, uid in enumerate(user_ids.cpu().numpy()):
                    if uid in user_item_history:
                        history = torch.tensor(user_item_history[uid], device=device)
                        predictions[i].index_fill_(0, history, float('-inf'))
            
            # Build vectorized relevant items for each unique user in the batch
            rel_lists = []
            for uid in user_ids.cpu().numpy():
                mask = ((users.cpu().numpy() == uid) & (ratings.cpu().numpy() > 0))
                rel = np.unique(items.cpu().numpy()[mask])
                rel_lists.append(rel.tolist())
            # Compute max relevant count for padding
            max_rel = max((len(r) for r in rel_lists), default=0)
            if max_rel == 0:
                continue  # skip users with no relevant items
            # Build padded relevant matrix and validity mask
            rel_matrix = torch.full((batch_size, max_rel), -1, device=device, dtype=torch.long)
            rel_mask = torch.zeros((batch_size, max_rel), device=device, dtype=torch.bool)
            rel_counts = torch.zeros(batch_size, device=device, dtype=torch.float)
            for i, rel in enumerate(rel_lists):
                if rel:
                    length = len(rel)
                    rel_matrix[i, :length] = torch.tensor(rel, device=device, dtype=torch.long)
                    rel_mask[i, :length] = True
                    rel_counts[i] = length

            # For each k, compute metrics vectorized over batch
            for k in k_list:
                # topk predictions: shape (batch_size, k)
                topk_vals, topk_idx = torch.topk(predictions, k, dim=1)
                # Update recommended item counts
                for i in range(batch_size):
                    item_recommended.index_add_(0, topk_idx[i], torch.ones(k, device=device))
                # Compute hit, recall, ndcg vectorized
                # Expand topk indices: shape (batch_size, k, 1)
                topk_exp = topk_idx.unsqueeze(2)
                # Compare with rel_matrix (batch_size, 1, max_rel)
                cmp = (topk_exp == rel_matrix.unsqueeze(1))  # shape (batch_size, k, max_rel)
                # Determine if each top-k item is relevant
                hit_matrix = cmp.any(dim=2)  # (batch_size, k), boolean
                # Hit: per user, did any topk item hit any relevant?
                hit = (hit_matrix.any(dim=1)).float()  # (batch_size,)
                # Recall: per user, count hits divided by rel_counts
                recall = (hit_matrix.float().sum(dim=1)) / (rel_counts + 1e-8)  # (batch_size,)
                # NDCG: discount vector: (k,)
                discounts = torch.tensor([1.0 / math.log2(i + 2) for i in range(k)], device=device)
                dcg = (hit_matrix.float() * discounts).sum(dim=1)
                # Ideal DCG: for each user, sum best discounts for min(rel_counts, k) items
                ideal_dcg = torch.zeros(batch_size, device=device)
                for i in range(batch_size):
                    cutoff = int(min(rel_counts[i].item(), k))
                    if cutoff > 0:
                        ideal_dcg[i] = discounts[:cutoff].sum()
                ndcg = torch.where(ideal_dcg > 0, dcg / ideal_dcg, torch.zeros_like(dcg))
                # Popularity: average popularity of topk items per user
                if item_popularity:
                    pop = []
                    for i in range(batch_size):
                        pop_vals = [item_popularity.get(int(item), 0) for item in topk_idx[i]]
                        pop.append(sum(pop_vals) / k)
                    avg_pop = torch.tensor(pop, device=device, dtype=torch.float)
                else:
                    avg_pop = torch.zeros(batch_size, device=device)
                
                # Sum metrics over batch
                metrics[k]['hit'] += hit.sum().item()
                metrics[k]['recall'] += recall.sum().item()
                metrics[k]['ndcg'] += ndcg.sum().item()
                metrics[k]['avg_popularity'] += avg_pop.sum().item()
                
    # After all batches, finalize metrics
    for k in k_list:
        for m in ['ndcg', 'recall', 'hit', 'avg_popularity']:
            metrics[k][m] /= user_total if user_total > 0 else 1
        # Coverage
        metrics[k]['coverage'] = (torch.count_nonzero(item_recommended).item() / n_items) if n_items > 0 else 0
        # Gini coefficient
        if n_items > 0:
            sorted_counts = torch.sort(item_recommended)[0]
            indices = torch.arange(1, n_items + 1, dtype=torch.float)
            gini = 1 - (2 * torch.sum((n_items + 1 - indices) * sorted_counts)) / (n_items * torch.sum(sorted_counts) + 1e-8)
            metrics[k]['gini'] = gini.item()
    
    return metrics

def calculate_ndcg(ranked_matrix, relevant_matrix, rel_mask, k):
    """
    Vectorized NDCG calculation.
    ranked_matrix: Tensor of shape (B, k) for ranked items.
    relevant_matrix: Tensor of shape (B, max_rel) padded with -1.
    rel_mask: Boolean tensor of same shape as relevant_matrix indicating valid entries.
    Returns: Tensor (B,) with ndcg for each user.
    """
    B = ranked_matrix.shape[0]
    discounts = torch.tensor([1.0 / math.log2(i + 2) for i in range(k)], device=ranked_matrix.device)
    topk_exp = ranked_matrix.unsqueeze(2)  # (B,k,1)
    cmp = (topk_exp == relevant_matrix.unsqueeze(1))  # (B,k, max_rel)
    rel_hits = cmp.any(dim=2).float()  # (B,k)
    dcg = (rel_hits * discounts).sum(dim=1)
    ideal_dcg = torch.zeros(B, device=ranked_matrix.device)
    # Compute ideal dcg per user
    rel_counts = rel_mask.float().sum(dim=1)
    for i in range(B):
        cutoff = int(min(rel_counts[i].item(), k))
        if cutoff > 0:
            ideal_dcg[i] = discounts[:cutoff].sum()
    ndcg = torch.where(ideal_dcg > 0, dcg / ideal_dcg, torch.zeros_like(dcg))
    return ndcg

def calculate_recall(ranked_matrix, relevant_matrix, rel_mask):
    """
    Vectorized Recall calculation.
    ranked_matrix: Tensor of shape (B, k)
    relevant_matrix: Tensor of shape (B, max_rel)
    rel_mask: Boolean tensor of shape (B, max_rel)
    Returns: Tensor (B,) of recall values.
    """
    B = ranked_matrix.shape[0]
    topk_exp = ranked_matrix.unsqueeze(2)
    cmp = (topk_exp == relevant_matrix.unsqueeze(1))
    hits = cmp.any(dim=2).float().sum(dim=1)
    rel_counts = rel_mask.float().sum(dim=1)
    recall = hits / (rel_counts + 1e-8)
    return recall

def calculate_hit_rate(ranked_matrix, relevant_matrix, rel_mask):
    """
    Vectorized Hit Rate calculation.
    Returns: Tensor (B,) of hit rate (1 if hit, else 0).
    """
    topk_exp = ranked_matrix.unsqueeze(2)
    cmp = (topk_exp == relevant_matrix.unsqueeze(1))
    hit = (cmp.any(dim=2).any(dim=1)).float()
    return hit

def precalculate_average_popularity(dataset):
    # calculate how many links each item has
    # count occurences of items with torch.bincount
    # return a dict with item_id as key and popularity as value
    item_popularity = torch.bincount(dataset.inter_feat['item_id'].values)
    item_popularity = item_popularity.cpu().numpy()
    item_popularity_dict = {i: item_popularity[i] for i in range(len(item_popularity))}
    return item_popularity_dict
