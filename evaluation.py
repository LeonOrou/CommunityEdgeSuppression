import numpy as np
import torch
import json
from collections import defaultdict


def evaluate_model(model, dataset, config, stage='cv', k_values=[10, 20, 50, 100]):
    """
    Evaluate model using the complete graph structure but excluding training interactions
    stage: 'full_train' for full training evaluation, 'loo' for leave-one-out evaluation
    k_values: list of k_values values to evaluate (e.g., [10, 20, 50])
    """
    model.eval()

    if isinstance(k_values, int):
        k_values = [k_values]

    k_values = sorted(k_values)
    max_k = max(k_values)

    encoded_to_genres = load_genre_mapping(dataset)

    if stage == 'full_train':
        dataset.val_df = dataset.test_df

    train_user_items = dataset.train_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

    # Initialize metric storage for each k_values
    metrics_by_k = {k_val: {
        'ndcg_scores': [],
        'recall_scores': [],
        'precision_scores': [],
        'mrr_scores': [],
        'hit_rate_scores': [],
        'all_recommended_items': set(),
        'simpson_scores': [],
        'intra_list_diversity_scores': [],
        'popularity_lift_scores': [],
        'avg_recommended_popularity_scores': [],
        'normalized_genre_entropy_scores': [],
        'unique_genres_count_scores': []
    } for k_val in k_values}

    # Store recommendation frequencies and global recommendations for calibration
    item_recommendation_freq_by_k = {k_val: defaultdict(int) for k_val in k_values}
    all_global_recommendations_by_k = {k_val: [] for k_val in k_values}

    # Store community distributions for each user for community bias calculation
    user_item_community_distributions = {k_val: [] for k_val in k_values}

    with torch.no_grad():
        user_emb, item_emb = model(dataset.complete_edge_index, dataset.current_edge_weight)

        user_test_items = dataset.val_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

        for user_id, true_items in user_test_items.items():
            if user_id >= dataset.num_users:
                continue

            user_embedding = user_emb[user_id:user_id + 1]
            scores = torch.matmul(user_embedding, item_emb.T).squeeze().cpu().numpy()

            # Exclude items that the user interacted with during training
            train_items = list(train_user_items[user_id])
            scores_filtered = scores.copy()
            scores_filtered[train_items] = float('-inf')

            top_max_k_items = np.argsort(scores_filtered)[::-1][:max_k]

            # Get full ranking for MRR calculation
            full_ranking = np.argsort(scores_filtered)[::-1]

            for k_val in k_values:
                top_k_items = top_max_k_items[:k_val]

                metrics_by_k[k_val]['all_recommended_items'].update(top_k_items)

                all_global_recommendations_by_k[k_val].extend(top_k_items)

                for item in top_k_items:
                    item_recommendation_freq_by_k[k_val][item] += 1

                # Calculate community distribution for this user's recommendations
                if hasattr(config, 'item_labels_matrix_mask'):
                    user_community_dist = calculate_user_item_community_distribution(
                        top_k_items.copy(), config.item_labels_matrix_mask, config.device
                    )
                    user_item_community_distributions[k_val].append(user_community_dist)

                ndcg = calculate_ndcg(true_items, scores_filtered, k_val)
                recall = calculate_recall(true_items, top_k_items, k_val)
                precision = calculate_precision(true_items, top_k_items, k_val)
                mrr = calculate_mrr(true_items, full_ranking)
                hit_rate = calculate_hit_rate(true_items, top_k_items, k_val)

                metrics_by_k[k_val]['ndcg_scores'].append(ndcg)
                metrics_by_k[k_val]['recall_scores'].append(recall)
                metrics_by_k[k_val]['precision_scores'].append(precision)
                metrics_by_k[k_val]['mrr_scores'].append(mrr)
                metrics_by_k[k_val]['hit_rate_scores'].append(hit_rate)

                additional_metrics = calculate_additional_metrics(
                    top_k_items, encoded_to_genres, dataset)

                metrics_by_k[k_val]['simpson_scores'].append(additional_metrics['simpson_index'])
                metrics_by_k[k_val]['intra_list_diversity_scores'].append(additional_metrics['intra_list_diversity'])
                metrics_by_k[k_val]['popularity_lift_scores'].append(additional_metrics['popularity_lift'])
                metrics_by_k[k_val]['avg_recommended_popularity_scores'].append(
                    additional_metrics['avg_recommended_popularity'])
                metrics_by_k[k_val]['normalized_genre_entropy_scores'].append(
                    additional_metrics['normalized_genre_entropy'])
                metrics_by_k[k_val]['unique_genres_count_scores'].append(additional_metrics['unique_genres_count'])

    results = {}
    for k_val in k_values:
        ndcg_scores = metrics_by_k[k_val]['ndcg_scores']
        recall_scores = metrics_by_k[k_val]['recall_scores']
        precision_scores = metrics_by_k[k_val]['precision_scores']
        mrr_scores = metrics_by_k[k_val]['mrr_scores']
        hit_rate_scores = metrics_by_k[k_val]['hit_rate_scores']

        simpson_scores = metrics_by_k[k_val]['simpson_scores']
        intra_list_diversity_scores = metrics_by_k[k_val]['intra_list_diversity_scores']
        popularity_lift_scores = metrics_by_k[k_val]['popularity_lift_scores']
        avg_recommended_popularity_scores = metrics_by_k[k_val]['avg_recommended_popularity_scores']
        normalized_genre_entropy_scores = metrics_by_k[k_val]['normalized_genre_entropy_scores']
        unique_genres_count_scores = metrics_by_k[k_val]['unique_genres_count_scores']

        all_recommended_items = metrics_by_k[k_val]['all_recommended_items']
        item_coverage = calculate_item_coverage(list(all_recommended_items), dataset.num_items)

        recommendation_counts = np.array(list(item_recommendation_freq_by_k[k_val].values()))
        if len(recommendation_counts) > 0:
            recommendation_probs = recommendation_counts / recommendation_counts.sum()
            gini_index = calculate_gini_coefficient(recommendation_probs)
        else:
            gini_index = 0.0

        popularity_calibration = calculate_popularity_calibration(
            all_global_recommendations_by_k[k_val], dataset
        )

        # Calculate community bias if community data is available
        user_community_bias = 0.0
        if (hasattr(config, 'item_labels_matrix_mask') and
                len(user_item_community_distributions[k_val]) > 0):
            # Stack all user community distributions
            all_user_community_dists = torch.stack(user_item_community_distributions[k_val])

            # Calculate community bias using the existing function
            user_bias, _ = get_community_bias(item_communities_each_user_dist=all_user_community_dists)
            user_community_bias = float(torch.mean(user_bias)) if user_bias is not None else None

        results[k_val] = {
            'NDCG': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'Recall': np.mean(recall_scores) if recall_scores else 0.0,
            'Precision': np.mean(precision_scores) if precision_scores else 0.0,
            'MRR': np.mean(mrr_scores) if mrr_scores else 0.0,
            'Hit Rate': np.mean(hit_rate_scores) if hit_rate_scores else 0.0,
            'Item Coverage': item_coverage,
            'Gini Index': gini_index,
            'Simpson Index Genre': np.mean(simpson_scores) if simpson_scores else 0.0,
            'Intra List Diversity': np.mean(intra_list_diversity_scores) if intra_list_diversity_scores else 0.0,
            'Popularity Lift': np.mean(popularity_lift_scores) if popularity_lift_scores else 0.0,
            'Avg Recommended Popularity': np.mean(
                avg_recommended_popularity_scores) if avg_recommended_popularity_scores else 0.0,
            'Normalized Genre Entropy': np.mean(
                normalized_genre_entropy_scores) if normalized_genre_entropy_scores else 0.0,
            'Unique Genres Count': np.mean(unique_genres_count_scores) if unique_genres_count_scores else 0.0,
            'Popularity Calibration': popularity_calibration,
            'User Community Bias': user_community_bias
        }

    return results


def evaluate_current_model_ndcg(model, dataset, k=10):
    """
    only get ndcg for early stopping, minimum calculations
    only necessary steps to calculate ndcg = calculate_ndcg(true_items, scores_filtered, k) for val_df
    """
    model.eval()

    ndcg_scores = []

    # Create a set of training interactions for each user to exclude from evaluation
    train_user_items = dataset.train_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

    with torch.no_grad():
        user_emb, item_emb = model(dataset.complete_edge_index, dataset.complete_edge_weight)

        user_val_items = dataset.val_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

        for user_id, true_items in user_val_items.items():
            if user_id >= dataset.num_users:
                continue

            user_embedding = user_emb[user_id:user_id + 1]
            scores = torch.matmul(user_embedding, item_emb.T).squeeze().cpu().numpy()

            # Exclude items that the user interacted with during training
            train_items = list(train_user_items[user_id])
            scores_filtered = scores.copy()
            scores_filtered[train_items] = float('-inf')

            ndcg = calculate_ndcg(true_items, scores_filtered, k)
            ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


# via euclidean distance between user/item community connectivity matrices and uniform distribution
# for each user and each item, the bias individually
def get_community_bias(item_communities_each_user_dist=None, user_communities_each_item_dist=None):
    """
    Get the community bias of the users and items.

    :param item_communities_each_user_dist: torch.tensor, item community distribution for each user
    :param user_communities_each_item_dist: torch.tensor, user community distribution for each item
    :return: tuple of torch.tensors, community bias for users and items
    """
    user_bias, item_bias = None, None

    if item_communities_each_user_dist is not None:
        uniform_distribution_users = torch.full_like(item_communities_each_user_dist, 1.0 / item_communities_each_user_dist.size(1))
        user_bias = torch.linalg.norm(uniform_distribution_users - item_communities_each_user_dist, dim=1)
        # Normalize the bias to be between 0 and 1
        # make worst possible distribution and divide by it to make it the maximum 1
        worst_distribution_users = torch.zeros_like(item_communities_each_user_dist)
        worst_distribution_users[:, 0] = 1.0  # worst distribution is all in one community
        # bias for each user and item, can be processed for distributions, averages, etc.
        bias_worst_users = torch.linalg.norm(uniform_distribution_users - worst_distribution_users, dim=1)
        user_bias /= bias_worst_users

    if user_communities_each_item_dist is not None:
        uniform_distribution_items = torch.full_like(user_communities_each_item_dist,
                                                     1.0 / user_communities_each_item_dist.size(1))
        item_bias = torch.linalg.norm(uniform_distribution_items - user_communities_each_item_dist, dim=1)
        worst_distribution_items = torch.zeros_like(user_communities_each_item_dist)
        worst_distribution_items[:, 0] = 1.0
        bias_worst_items = torch.linalg.norm(uniform_distribution_items - worst_distribution_items, dim=1)
        item_bias /= bias_worst_items

    return user_bias, item_bias


def print_metric_results(metrics, title="Results"):
    """Print metrics in a formatted table with k values as columns and metrics as rows."""
    if not metrics:
        print(f"No {title.lower()} available")
        return

    # Get available k values from metrics keys
    k_values = sorted(metrics.keys())

    # Get all metric names from the first k value
    metric_names = list(metrics[k_values[0]].keys())

    print(f"\n{title}")
    print("=" * 85)

    # Create header
    header = f"{'Metric':<20}"
    for k in k_values:
        header += f"{'k=' + str(k):>12}"
    print(header)
    print("-" * (20 + 12 * len(k_values)))

    # Print each metric row
    for metric_name in metric_names:
        row = f"{metric_name:<25}"
        for k in k_values:
            value = metrics[k][metric_name]
            row += f"{value:>12.4f}"
        print(row)


def calculate_user_item_community_distribution(recommended_items, item_labels_matrix_mask, device):
    """
    Calculate the distribution of item communities for a user's recommended items.

    Args:
        recommended_items: list/array of recommended item IDs (encoded)
        item_labels_matrix_mask: torch.tensor of shape (n_items, n_communities)
                                where entry [i,j] = 1 if item i belongs to community j
        device: torch device

    Returns:
        torch.tensor: normalized distribution over item communities for this user
    """
    if len(recommended_items) == 0:
        # Return uniform distribution if no recommendations
        n_communities = item_labels_matrix_mask.shape[1]
        return torch.ones(n_communities, device=device) / n_communities

    # Convert to tensor if needed
    if not isinstance(recommended_items, torch.Tensor):
        recommended_items = torch.tensor(recommended_items, device=device)

    # Ensure item_labels_matrix_mask is on the correct device
    if item_labels_matrix_mask.device != device:
        item_labels_matrix_mask = item_labels_matrix_mask.to(device)

    # Filter out items that are outside the matrix bounds
    valid_items = recommended_items[recommended_items < item_labels_matrix_mask.shape[0]]

    if len(valid_items) == 0:
        # Return uniform distribution if no valid items
        n_communities = item_labels_matrix_mask.shape[1]
        return torch.ones(n_communities, device=device) / n_communities

    # Get community memberships for recommended items
    item_community_memberships = item_labels_matrix_mask[valid_items]  # shape: (n_recommended_items, n_communities)

    # Sum across items to get total community counts
    community_counts = torch.sum(item_community_memberships, dim=0).float()  # shape: (n_communities,)

    # Normalize to get distribution
    total_count = torch.sum(community_counts)
    if total_count > 0:
        community_distribution = community_counts / total_count
    else:
        # Return uniform distribution if no community memberships
        n_communities = item_labels_matrix_mask.shape[1]
        community_distribution = torch.ones(n_communities, device=device) / n_communities

    return community_distribution


def load_genre_mapping(dataset):
    """
    Load genre labels and create mapping from encoded item IDs to genres
    """
    # Load genre labels from JSON file
    with open(f'dataset/{dataset.name}/saved/item_genre_labels_{dataset.name}.json', 'r') as f:
        item_genre_labels = json.load(f)

    # Create mapping from encoded item ID to genres
    # Need to map: original_item_id -> encoded_item_id -> genres
    item_id_to_encoded = dataset.complete_df.set_index('item_id')['item_encoded'].to_dict()

    item_id_to_encoded = {str(k): v for k, v in item_id_to_encoded.items()}

    encoded_to_genres = {}
    for original_item_id, genres in item_genre_labels.items():
        if original_item_id in item_id_to_encoded:
            encoded_id = item_id_to_encoded[original_item_id]
            encoded_to_genres[encoded_id] = genres

    return encoded_to_genres


def calculate_simpson_index_recommendations(recommended_items, encoded_to_genres):
    """
    Calculate Simpson Index for recommendation diversity based on genre distribution

    Args:
        recommended_items: list of recommended item IDs (encoded)
        encoded_to_genres: dict mapping encoded item ID to list of genre IDs

    Returns:
        float: Simpson Index (higher = more diverse)
    """
    if len(recommended_items) == 0:
        return 0.0

    # Count genre frequencies across all recommended items
    genre_counts = defaultdict(int)
    total_genre_assignments = 0

    for item_id in recommended_items:
        if item_id in encoded_to_genres:
            genres = encoded_to_genres[item_id]
            for genre in genres:
                genre_counts[genre] += 1
                total_genre_assignments += 1

    if total_genre_assignments == 0:
        return 0.0

    # Calculate Simpson Index: 1 - sum(p_i^2)
    # where p_i is the proportion of genre i
    simpson_index = 1.0
    for count in genre_counts.values():
        proportion = count / total_genre_assignments
        simpson_index -= proportion ** 2

    return simpson_index


def calculate_intra_list_diversity_jaccard(recommended_items, encoded_to_genres):
    """
    Calculate intra-list diversity using average pairwise Jaccard similarity
    Lower similarity = higher diversity

    Args:
        recommended_items: list of recommended item IDs (encoded)
        encoded_to_genres: dict mapping encoded item ID to list of genre IDs

    Returns:
        float: Average pairwise Jaccard similarity (lower = more diverse)
    """
    if len(recommended_items) < 2:
        return 0.0

    # Filter items that have genre information
    items_with_genres = []
    for item_id in recommended_items:
        if item_id in encoded_to_genres:
            items_with_genres.append((item_id, set(encoded_to_genres[item_id])))

    if len(items_with_genres) < 2:
        return 0.0

    # Calculate pairwise Jaccard similarities
    similarities = []
    n_items = len(items_with_genres)

    for i in range(n_items):
        for j in range(i + 1, n_items):
            genres_i = items_with_genres[i][1]
            genres_j = items_with_genres[j][1]

            intersection = len(genres_i.intersection(genres_j))
            union = len(genres_i.union(genres_j))

            if union > 0:
                jaccard_sim = intersection / union
                similarities.append(jaccard_sim)

    if len(similarities) == 0:
        return 0.0

    return np.mean(similarities)


def calculate_popularity_lift(recommended_items, dataset, baseline_method='catalog_average'):
    """
    Calculate popularity lift/calibration metrics

    Args:
        recommended_items: list of recommended item IDs (encoded)
        dataset: dataset object with item_popularities
        baseline_method: 'catalog_average' or 'user_profile'

    Returns:
        dict: containing popularity_lift, avg_recommended_popularity, baseline_popularity
    """
    if len(recommended_items) == 0:
        return {
            'popularity_lift': 0.0,
            'avg_recommended_popularity': 0.0,
            'baseline_popularity': 0.0
        }

    # Get popularities of recommended items
    recommended_popularities = []
    for item_id in recommended_items:
        if item_id < len(dataset.item_popularities):
            recommended_popularities.append(dataset.item_popularities[item_id])

    if len(recommended_popularities) == 0:
        return {
            'popularity_lift': 0.0,
            'avg_recommended_popularity': 0.0,
            'baseline_popularity': 0.0
        }

    avg_recommended_popularity = np.mean(recommended_popularities)

    # Calculate baseline popularity (catalog average)
    if baseline_method == 'catalog_average':
        baseline_popularity = np.mean(dataset.item_popularities)
    else:
        # Could implement user-specific baseline here if needed
        baseline_popularity = np.mean(dataset.item_popularities)

    # Calculate lift
    if baseline_popularity > 0:
        popularity_lift = avg_recommended_popularity / baseline_popularity
    else:
        popularity_lift = 1.0

    return {
        'popularity_lift': popularity_lift,
        'avg_recommended_popularity': avg_recommended_popularity,
        'baseline_popularity': baseline_popularity
    }


def calculate_popularity_calibration(all_recommended_items, dataset, num_bins=10):
    """
    Calculate popularity calibration - how well recommendation popularity distribution
    matches the catalog popularity distribution

    Args:
        all_recommended_items: list of all recommended items across all users (encoded)
        dataset: dataset object with item_popularities
        num_bins: number of popularity bins to use

    Returns:
        float: KL divergence between recommendation and catalog popularity distributions
               (lower = better calibration)
    """
    if len(all_recommended_items) == 0:
        return float('inf')

    # Get popularities
    catalog_popularities = dataset.item_popularities
    recommended_popularities = []

    for item_id in all_recommended_items:
        if item_id < len(dataset.item_popularities):
            recommended_popularities.append(dataset.item_popularities[item_id])

    if len(recommended_popularities) == 0:
        return float('inf')

    # Create bins based on catalog popularity distribution
    min_pop = np.min(catalog_popularities)
    max_pop = np.max(catalog_popularities)

    if min_pop == max_pop:
        return 0.0

    bins = np.linspace(min_pop, max_pop, num_bins + 1)

    # Calculate distributions
    catalog_hist, _ = np.histogram(catalog_popularities, bins=bins, density=True)
    recommended_hist, _ = np.histogram(recommended_popularities, bins=bins, density=True)

    # Normalize to get probabilities
    catalog_dist = catalog_hist / np.sum(catalog_hist) if np.sum(catalog_hist) > 0 else np.ones(num_bins) / num_bins
    recommended_dist = recommended_hist / np.sum(recommended_hist) if np.sum(recommended_hist) > 0 else np.ones(
        num_bins) / num_bins

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    catalog_dist = np.maximum(catalog_dist, epsilon)
    recommended_dist = np.maximum(recommended_dist, epsilon)

    # Calculate KL divergence
    kl_divergence = np.sum(recommended_dist * np.log(recommended_dist / catalog_dist))

    return kl_divergence


def calculate_normalized_genre_entropy(recommended_items, encoded_to_genres):
    """
    Calculate normalized entropy of recommended genre labels
    For each item, divide by number of labels, then calculate Shannon entropy

    Args:
        recommended_items: list of recommended item IDs (encoded)
        encoded_to_genres: dict mapping encoded item ID to list of genre IDs

    Returns:
        float: Normalized entropy of genre distribution
    """
    if len(recommended_items) == 0:
        return 0.0

    # Collect genre weights (1/num_labels per item for each genre)
    genre_weights = defaultdict(float)

    for item_id in recommended_items:
        if item_id in encoded_to_genres:
            genres = encoded_to_genres[item_id]
            if len(genres) > 0:
                weight_per_genre = 1.0 / len(genres)  # Normalize by number of labels per item
                for genre in genres:
                    genre_weights[genre] += weight_per_genre

    if len(genre_weights) == 0:
        return 0.0

    # Convert to probability distribution
    total_weight = sum(genre_weights.values())
    if total_weight == 0:
        return 0.0

    genre_probs = np.array([weight / total_weight for weight in genre_weights.values()])

    # Calculate Shannon entropy
    # Entropy = -sum(p * log2(p))
    entropy = 0.0
    for prob in genre_probs:
        if prob > 0:
            entropy -= prob * np.log2(prob)

    # Normalize by maximum possible entropy (log2 of number of unique genres)
    max_entropy = np.log2(len(genre_probs)) if len(genre_probs) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return normalized_entropy


def calculate_unique_genres_count(recommended_items, encoded_to_genres):
    """
    Count number of unique genres recommended to a user

    Args:
        recommended_items: list of recommended item IDs (encoded)
        encoded_to_genres: dict mapping encoded item ID to list of genre IDs

    Returns:
        int: Number of unique genres in the recommendation list
    """
    if len(recommended_items) == 0:
        return 0

    unique_genres = set()

    for item_id in recommended_items:
        if item_id in encoded_to_genres:
            genres = encoded_to_genres[item_id]
            unique_genres.update(genres)

    return len(unique_genres)


# Integration function to modify your evaluate_model function
def calculate_additional_metrics(top_k_items, encoded_to_genres, dataset, all_recommended_items_global=None):
    """
    Calculate all additional metrics for a single recommendation list

    Args:
        top_k_items: list of recommended items for this user
        encoded_to_genres: genre mapping
        dataset: dataset object
        all_recommended_items_global: global list of all recommendations (for calibration)

    Returns:
        dict: containing all calculated metrics
    """
    metrics = {}

    # Simpson Index
    metrics['simpson_index'] = calculate_simpson_index_recommendations(top_k_items, encoded_to_genres)

    # Intra-list Diversity (average pairwise Jaccard similarity)
    metrics['intra_list_diversity'] = calculate_intra_list_diversity_jaccard(top_k_items, encoded_to_genres)

    # Popularity Lift
    popularity_metrics = calculate_popularity_lift(top_k_items, dataset)
    metrics.update(popularity_metrics)

    # Normalized Genre Entropy
    metrics['normalized_genre_entropy'] = calculate_normalized_genre_entropy(top_k_items, encoded_to_genres)

    # Unique Genres Count
    metrics['unique_genres_count'] = calculate_unique_genres_count(top_k_items, encoded_to_genres)

    # Popularity Calibration (if global recommendations provided)
    if all_recommended_items_global is not None:
        metrics['popularity_calibration'] = calculate_popularity_calibration(all_recommended_items_global, dataset)

    return metrics


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


