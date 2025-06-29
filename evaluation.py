import numpy as np
import pandas as pd
import torch
import json
from collections import defaultdict
from scipy.sparse import csr_matrix
from itertools import combinations


def evaluate_model(model, dataset, config, stage='cv', k_values=[10, 20, 50, 100]):
    """
    Evaluate model using the complete graph structure but excluding training interactions.
    Adapted for different model types: LightGCN, MultiVAE, or ItemKNN.
    stage: 'full_train' for full training evaluation, 'cv' for cross-validation evaluation
    k_values: list of k_values values to evaluate (e.g., [10, 20, 50])
    """
    if config.model_name != 'ItemKNN':
        model.eval()

    if isinstance(k_values, int):
        k_values = [k_values]

    k_values = sorted(k_values)
    max_k = max(k_values)

    encoded_to_genres = load_genre_mapping(dataset)

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
        'avg_rec_popularity_scores': [],
        'normalized_genre_entropy_scores': [],
        'unique_genres_count_scores': []
    } for k_val in k_values}

    # Store recommendation frequencies and global recommendations for calibration
    item_recommendation_freq_by_k = {k_val: defaultdict(int) for k_val in k_values}
    all_global_recommendations_by_k = {k_val: [] for k_val in k_values}

    # Store community distributions for each user for community bias calculation
    user_item_community_distributions = {k_val: [] for k_val in k_values}

    user_test_items = dataset.val_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

    with torch.no_grad():
        if config.model_name == 'LightGCN':
            # LightGCN evaluation
            user_emb, item_emb = model(dataset.complete_edge_index, dataset.complete_edge_weight)

            for user_id, true_items in user_test_items.items():
                if user_id >= dataset.num_users:
                    continue

                user_embedding = user_emb[user_id:user_id + 1]
                scores = torch.matmul(user_embedding, item_emb.T).squeeze().cpu().numpy()

                # Exclude items that the user interacted with during training
                train_items = list(train_user_items.get(user_id, []))
                scores_filtered = scores.copy()
                scores_filtered[train_items] = float('-inf')

                process_user_recommendations(
                    user_id, true_items, scores_filtered, train_items, max_k, k_values,
                    metrics_by_k, item_recommendation_freq_by_k, all_global_recommendations_by_k,
                    user_item_community_distributions, config, encoded_to_genres, dataset
                )

        elif config.model_name == 'MultiVAE':
            train_matrix = csr_matrix(
                (dataset.train_df['rating'].values,
                 (dataset.train_df['user_encoded'].values, dataset.train_df['item_encoded'].values)),
                shape=(dataset.num_users, dataset.num_items),
                dtype=np.float32)

            device = next(model.parameters()).device
            user_ids = list(user_test_items.keys())
            user_input = torch.tensor(train_matrix[user_ids].toarray(), device=device, dtype=torch.float32)

            recon, _, _ = model(user_input)
            scores_batch = recon.cpu().numpy()  # Shape: (num_valid_users, num_items)

            # Create mask for training items to exclude
            scores_filtered = scores_batch.copy()
            for i, user_id in enumerate(user_test_items.keys()):
                train_items = list(train_user_items.get(user_id, []))
                if train_items:
                    scores_filtered[i, train_items] = float('-inf')
                process_user_recommendations(
                    user_id, user_test_items[user_id], scores_filtered[i], train_items, max_k, k_values,
                    metrics_by_k, item_recommendation_freq_by_k, all_global_recommendations_by_k,
                    user_item_community_distributions, config, encoded_to_genres, dataset
                )

        elif config.model_name == 'ItemKNN':
            # ItemKNN evaluation
            for user_id, true_items in user_test_items.items():
                # if user_id >= dataset.num_users:
                #     continue

                # Get recommendations from ItemKNN
                recommendations = model.predict(user_id, n_items=dataset.num_items)

                # ItemKNN returns [(item_id, score), ...] for each user
                if len(recommendations) > 0 and len(recommendations[0]) > 0:
                    # Extract scores and create score array
                    scores_filtered = np.full(dataset.num_items, float('-inf'))
                    for item_id, score in recommendations[0]:
                        if 0 <= item_id < dataset.num_items:
                            scores_filtered[int(item_id)] = score

                    # Training items are already excluded by ItemKNN internally
                    train_items = list(train_user_items.get(user_id, []))

                process_user_recommendations(
                    user_id, true_items, scores_filtered, train_items, max_k, k_values,
                    metrics_by_k, item_recommendation_freq_by_k, all_global_recommendations_by_k,
                    user_item_community_distributions, config, encoded_to_genres, dataset
                )

        else:
            raise ValueError(f"Unsupported model type: {config.model_name}")

    # Calculate final results
    results = _calculate_final_results(
        k_values, metrics_by_k, item_recommendation_freq_by_k,
        all_global_recommendations_by_k, user_item_community_distributions,
        config, dataset
    )

    return results


def process_user_recommendations(user_id, true_items, scores_filtered, train_items, max_k, k_values,
                                 metrics_by_k, item_recommendation_freq_by_k, all_global_recommendations_by_k,
                                 user_item_community_distributions, config, encoded_to_genres, dataset):
    """
    Process recommendations for a single user across all k values.
    """
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

        # Calculate standard metrics
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

        # Calculate additional metrics
        additional_metrics = calculate_additional_metrics(
            top_k_items, encoded_to_genres, dataset)

        metrics_by_k[k_val]['simpson_scores'].append(additional_metrics['simpson_index'])
        metrics_by_k[k_val]['intra_list_diversity_scores'].append(additional_metrics['intra_list_diversity'])
        metrics_by_k[k_val]['popularity_lift_scores'].append(additional_metrics['popularity_lift'])
        metrics_by_k[k_val]['avg_rec_popularity_scores'].append(
            additional_metrics['avg_rec_popularity'])
        metrics_by_k[k_val]['normalized_genre_entropy_scores'].append(
            additional_metrics['normalized_genre_entropy'])
        metrics_by_k[k_val]['unique_genres_count_scores'].append(additional_metrics['unique_genres_count'])


def _calculate_final_results(k_values, metrics_by_k, item_recommendation_freq_by_k,
                             all_global_recommendations_by_k, user_item_community_distributions,
                             config, dataset):
    """
    Calculate final aggregated results for all k values.
    """
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
        avg_rec_popularity_scores = metrics_by_k[k_val]['avg_rec_popularity_scores']
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

        pop_miscalibration = popularity_miscalibration(
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
            'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'recall': np.mean(recall_scores) if recall_scores else 0.0,
            'precision': np.mean(precision_scores) if precision_scores else 0.0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'hit_rate': np.mean(hit_rate_scores) if hit_rate_scores else 0.0,
            'item_coverage': item_coverage,
            'gini_index': gini_index,
            'simpson_index_genre': np.mean(simpson_scores) if simpson_scores else 0.0,
            'intra_list_diversity': np.mean(intra_list_diversity_scores) if intra_list_diversity_scores else 0.0,
            'popularity_lift': np.mean(popularity_lift_scores) if popularity_lift_scores else 0.0,
            'avg_rec_popularity': np.mean(
                avg_rec_popularity_scores) if avg_rec_popularity_scores else 0.0,
            'normalized_genre_entropy': np.mean(
                normalized_genre_entropy_scores) if normalized_genre_entropy_scores else 0.0,
            'unique_genres_count': np.mean(unique_genres_count_scores) if unique_genres_count_scores else 0.0,
            'pop_miscalibration': pop_miscalibration,
            'user_community_bias': user_community_bias
        }

    return results


def evaluate_current_model_ndcg(model, dataset, model_type='LightGCN', k=10):
    """
    Adapted NDCG evaluation for different model types: LightGCN, MultiVAE, or ItemKNN.
    Only calculates NDCG for early stopping with minimum calculations.
    """
    if model_type != 'ItemKNN':
        model.eval()
    ndcg_scores = []

    # Create a set of training interactions for each user to exclude from evaluation
    train_user_items = dataset.train_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()
    user_val_items = dataset.val_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

    with torch.no_grad():
        if model_type == 'LightGCN':
            # dataset.val_edge_index, dataset.val_edge_weight = dataset.create_bipartite_graph(df=dataset.val_df)
            user_emb, item_emb = model(dataset.complete_edge_index, dataset.complete_edge_weight)
            for user_id, true_items in user_val_items.items():
                if user_id >= dataset.num_users:
                    continue

                user_embedding = user_emb[user_id]
                scores = torch.matmul(user_embedding, item_emb.T).squeeze().cpu().numpy()

                # Exclude items that the user interacted with during training
                train_items = list(train_user_items.get(user_id, []))
                scores_filtered = scores.copy()
                scores_filtered[train_items] = float('-inf')

                ndcg = calculate_ndcg(true_items, scores_filtered, k)
                ndcg_scores.append(ndcg)

        elif model_type == 'MultiVAE':
            train_matrix = csr_matrix(
                (dataset.train_df['rating'].values, (dataset.train_df['user_encoded'].values, dataset.train_df['item_encoded'].values)),
                shape=(dataset.num_users, dataset.num_items),
                dtype=np.float32)

            device = next(model.parameters()).device
            user_ids = list(user_val_items.keys())
            user_rows = train_matrix[user_ids].toarray()  # Shape: (num_valid_users, num_items)
            user_input = torch.tensor(user_rows, device=device, dtype=torch.float32)

            # Get recommendations for all users at once
            recon, _, _ = model(user_input)
            scores_batch = recon.cpu().numpy()  # Shape: (num_valid_users, num_items)

            # Create mask for training items to exclude
            scores_filtered = scores_batch.copy()
            for i, user_id in enumerate(user_ids):
                train_items = list(train_user_items.get(user_id, []))
                if train_items:
                    scores_filtered[i, train_items] = float('-inf')

            # Calculate NDCG for all users (assuming calculate_ndcg can handle batched input)
            # If calculate_ndcg can't handle batched input, use list comprehension:
            ndcg_scores = [
                calculate_ndcg(user_val_items[user_id], scores_filtered[i], k)
                for i, user_id in enumerate(user_val_items.keys())
            ]

        elif model_type == 'ItemKNN':
            for user_id, true_items in user_val_items.items():
                # if user_id >= dataset.num_users:
                #     continue

                recommendations = model.predict(user_id, n_items=k)

                # ItemKNN returns [(item_id, score), ...] for each user
                if len(recommendations) > 0 and len(recommendations[0]) > 0:
                    # Extract scores and create score array
                    scores = np.full(dataset.num_items, float('-inf'))
                    for item_id, score in recommendations[0]:  # [0] as its only one user
                        if 0 <= item_id < dataset.num_items:
                            scores[int(item_id)] = score

                ndcg = calculate_ndcg(true_items, scores, k)
                ndcg_scores.append(ndcg)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


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
    header = f"{'Metric':<25}"
    for k in k_values:
        header += f"{'k=' + str(k):>12}"
    print(header)
    print("-" * (20 + 12 * len(k_values)))

    for metric_name in metric_names:
        row = f"{metric_name:<25}"
        for k in k_values:
            value = metrics[k][metric_name]
            if not isinstance(value, float):
                print(value)
                continue
            row += f"{np.round(value, 4):>12}"
        print(row)


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
            # Normalize by number of genres to avoid bias towards items with more genres
            # weight = 1.0 / len(genres)
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


def calculate_intra_list_diversity_genre(top_k_items, encoded_to_genres):
    """
    Calculate intra-list diversity using average pairwise Jaccard distance.

    For each pair of items in the recommendation list, calculate Jaccard similarity
    of their genre sets, then average across all pairs. Higher similarity means
    lower diversity, so we can report 1 - avg_similarity as diversity.

    Higher values indicate more diverse recommendations.
        top_k_items: list of recommended item IDs
        encoded_to_genres: dict mapping item_id -> list of genre_ids

    Returns:
        float: Average pairwise Jaccard distance (0 to 1)
    """
    if len(top_k_items) < 2:
        return 0.0  # Need at least 2 items to calculate diversity

    items_with_genres = []
    for item_id in top_k_items:
        if item_id in encoded_to_genres:
            genres = encoded_to_genres[item_id]
            if len(genres) > 0:
                items_with_genres.append((item_id, set(genres)))

    if len(items_with_genres) < 2:
        return 0.0  # Need at least 2 items with genres

    similarities = []
    n_items = len(items_with_genres)

    for i in range(n_items):
        for j in range(i + 1, n_items):
            genres_i = items_with_genres[i][1]
            genres_j = items_with_genres[j][1]

            # Jaccard similarity = |intersection| / |union|
            intersection = len(genres_i.intersection(genres_j))
            union = len(genres_i.union(genres_j))

            if union > 0:
                jaccard_similarity = intersection / union
                similarities.append(jaccard_similarity)

    if len(similarities) == 0:
        return 0.0

    avg_similarity = np.mean(similarities)
    # Convert to diversity: higher diversity = lower similarity
    # Jaccard distance = 1 - Jaccard similarity
    diversity = 1.0 - avg_similarity

    return diversity


def calculate_intra_list_diversity(top_k_items, dataset):
    '''
    Calculate intra-list diversity using average pairwise Jaccard distance between items

    :param top_k_items: list of top-k recommended item IDs
    :param dataset: dataset object with complete_df containing ['users_encoded', 'item_encoded', 'rating']
    :return: diversity: float (average pairwise Jaccard distance)
    '''

    if len(top_k_items) < 2:
        return 0.0  # No diversity possible with less than 2 items

    # Get the complete dataframe with user-item interactions
    complete_df = dataset.complete_df

    # Create item profiles (sets of users who interacted with each item)
    item_profiles = {}

    for item_id in top_k_items:
        # Get all users who interacted with this item
        users_for_item = complete_df[complete_df['item_encoded'] == item_id]['user_encoded'].unique()
        item_profiles[item_id] = set(users_for_item)

    # Calculate pairwise Jaccard distances
    jaccard_distances = []

    # Generate all pairs of items in the recommendation list
    for item1, item2 in combinations(top_k_items, 2):
        profile1 = item_profiles[item1]
        profile2 = item_profiles[item2]

        # Calculate Jaccard similarity
        intersection = len(profile1.intersection(profile2))
        union = len(profile1.union(profile2))

        if union == 0:
            jaccard_similarity = 0.0
        else:
            jaccard_similarity = intersection / union

        # Jaccard distance = 1 - Jaccard similarity
        jaccard_distance = 1.0 - jaccard_similarity
        jaccard_distances.append(jaccard_distance)

    # Return average pairwise Jaccard distance
    diversity = np.mean(jaccard_distances)

    return diversity


def calculate_popularity_lift(recommended_items, dataset, baseline_method='catalog_average'):
    """
    Calculate popularity lift/calibration metrics

    Args:
        recommended_items: list of recommended item IDs (encoded)
        dataset: dataset object with item_popularities
        baseline_method: 'catalog_average' or 'user_profile'

    Returns:
        dict: containing popularity_lift, avg_rec_popularity, baseline_popularity
    """
    if len(recommended_items) == 0:
        return {
            'popularity_lift': 0.0,
            'avg_rec_popularity': 0.0,
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
            'avg_rec_popularity': 0.0,
            'baseline_popularity': 0.0
        }

    avg_rec_popularity = np.mean(recommended_popularities)

    # Calculate baseline popularity (catalog average)
    if baseline_method == 'catalog_average':
        baseline_popularity = np.mean(dataset.item_popularities.array)
    else:
        # Could implement user-specific baseline here if needed
        baseline_popularity = np.mean(dataset.item_popularities.array)

    # Calculate lift
    if baseline_popularity > 0:
        popularity_lift = avg_rec_popularity / baseline_popularity
    else:
        popularity_lift = 1.0

    return {
        'popularity_lift': popularity_lift,
        'avg_rec_popularity': avg_rec_popularity,
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
    catalog_popularities = dataset.item_popularities.array
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

    # Normalize to get probability distributions
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


def calculate_normalized_genre_entropy(top_k_items, encoded_to_genres):
    """
    Calculate normalized entropy of genre distribution in recommendations.

    For each item with multiple genres, each genre gets weight 1/num_genres_for_that_item.
    Then calculate Shannon entropy of the resulting genre distribution and normalize
    by maximum possible entropy.

    Higher values indicate more diverse genre distribution.

    Args:
        top_k_items: list of recommended item IDs
        encoded_to_genres: dict mapping item_id -> list of genre_ids

    Returns:
        float: Normalized genre entropy (0 to 1)
    """
    if len(top_k_items) == 0:
        return 0.0

    # Calculate weighted genre distribution
    genre_weights = defaultdict(float)

    for item_id in top_k_items:
        if item_id in encoded_to_genres:
            genres = encoded_to_genres[item_id]
            if len(genres) > 0:
                # Each genre gets weight 1/num_genres for this item
                weight_per_genre = 1.0 / len(genres)
                for genre in genres:
                    genre_weights[genre] += weight_per_genre

    if len(genre_weights) == 0:
        return 0.0

    # Convert to probability distribution
    total_weight = sum(genre_weights.values())
    if total_weight == 0:
        return 0.0

    genre_probs = np.array([weight / total_weight for weight in genre_weights.values()])

    # Calculate Shannon entropy: H = -sum(p * log2(p))
    entropy = 0.0
    for prob in genre_probs:
        if prob > 0:
            entropy -= prob * np.log2(prob)

    # Normalize by maximum possible entropy (log2 of number of unique genres)
    num_unique_genres = len(genre_weights)
    if num_unique_genres <= 1:
        return 0.0  # No diversity possible with 0 or 1 genre

    max_entropy = np.log2(num_unique_genres)
    normalized_entropy = entropy / max_entropy

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

    # Intra-list Diversity (average pairwise Jaccard similarity of items)
    metrics['intra_list_diversity'] = calculate_intra_list_diversity_genre(top_k_items, encoded_to_genres)

    # Popularity Lift
    popularity_metrics = calculate_popularity_lift(top_k_items, dataset)
    metrics.update(popularity_metrics)

    # Normalized Genre Entropy
    metrics['normalized_genre_entropy'] = calculate_normalized_genre_entropy(top_k_items, encoded_to_genres)

    # Unique Genres Count
    metrics['unique_genres_count'] = calculate_unique_genres_count(top_k_items, encoded_to_genres)

    # Popularity Calibration (if global recommendations provided)
    if all_recommended_items_global is not None:
        metrics['pop_miscalibration'] = popularity_miscalibration(all_recommended_items_global, dataset)

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


def calculate_gini_coefficient(rel_rec_freqs):
    """
    Calculate Gini coefficient for recommendation distribution
    Higher values indicate more inequality (few items get most recommendations)
    Lower values indicate more equality (recommendations spread evenly)
    """
    if len(rel_rec_freqs) == 0:
        return 0.0

    # Sort rel_rec_freqs
    sorted_probs = np.sort(rel_rec_freqs)
    n = len(sorted_probs)

    # Calculate Gini coefficient
    index = np.arange(1, n + 1)
    gini = np.sum((2 * index - n - 1) * sorted_probs) / (n-1)

    return gini

###############################################
# vectorized
###############
import numpy as np
import torch
import json
from collections import defaultdict
from scipy.sparse import csr_matrix, coo_matrix
import pandas as pd


def evaluate_model_vectorized(model, dataset, config, k_values=[10, 20, 50, 100]):
    """
    Memory-efficient vectorized evaluation using sparse representations.
    """
    if config.model_name != 'ItemKNN':
        model.eval()

    if isinstance(k_values, int):
        k_values = [k_values]

    k_values = sorted(k_values)
    max_k = max(k_values)

    encoded_to_genres = load_genre_mapping(dataset)

    # Prepare sparse data structures
    train_user_items = dataset.train_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()
    user_test_items = dataset.val_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

    # Get valid user IDs
    valid_user_ids = [uid for uid in user_test_items.keys() if uid < dataset.num_users]
    valid_user_ids = np.array(valid_user_ids)

    # Create sparse relevance matrix
    relevance_sparse = create_sparse_relevance_matrix(valid_user_ids, user_test_items, dataset.num_items)

    # Process evaluation in batches to manage memory
    batch_size = min(2048, len(valid_user_ids))  # Adaptive batch size
    results = {}

    for k_val in k_values:
        results[k_val] = {
            'ndcg_scores': [],
            'recall_scores': [],
            'precision_scores': [],
            'mrr_scores': [],
            'hit_rate_scores': [],
            'all_recommended_items': set(),
            'simpson_scores': [],
            'intra_list_diversity_scores': [],
            'popularity_lift_scores': [],
            'avg_rec_popularity_scores': [],
            'normalized_genre_entropy_scores': [],
            'unique_genres_count_scores': [],
            'item_recommendation_freq': defaultdict(int),
            'all_global_recommendations': [],
            'user_community_distributions': []
        }

    # Process in batches
    for batch_start in range(0, len(valid_user_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_user_ids))
        batch_user_ids = valid_user_ids[batch_start:batch_end]

        # Get scores for this batch
        batch_scores = get_batch_user_scores(model, dataset, config, batch_user_ids, train_user_items)

        # Get batch relevance data
        batch_relevance = relevance_sparse[batch_start:batch_end]

        # Process this batch
        process_batch_metrics(
            batch_scores, batch_relevance, batch_user_ids, user_test_items,
            k_values, max_k, results, encoded_to_genres, dataset, config
        )

    # Calculate final aggregated results
    final_results = aggregate_batch_results(results, k_values, dataset)

    return final_results


def create_sparse_relevance_matrix(valid_user_ids, user_test_items, num_items):
    """
    Create sparse relevance matrix using COO format for memory efficiency.
    """
    row_indices = []
    col_indices = []

    user_id_to_row = {uid: i for i, uid in enumerate(valid_user_ids)}

    for user_id, test_items in user_test_items.items():
        if user_id in user_id_to_row:
            row_idx = user_id_to_row[user_id]
            for item_id in test_items:
                row_indices.append(row_idx)
                col_indices.append(item_id)

    # Create sparse matrix
    data = np.ones(len(row_indices), dtype=bool)
    relevance_sparse = coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(valid_user_ids), num_items),
        dtype=bool
    ).tocsr()  # Convert to CSR for efficient row slicing

    return relevance_sparse


def get_batch_user_scores(model, dataset, config, batch_user_ids, train_user_items):
    """
    Get prediction scores for a batch of users, excluding training items.
    """
    batch_size = len(batch_user_ids)

    with torch.no_grad():
        if config.model_name == 'LightGCN':
            user_emb, item_emb = model(dataset.complete_edge_index, dataset.complete_edge_weight)

            # Get embeddings for batch users only
            batch_user_indices = torch.tensor(batch_user_ids, device=user_emb.device)
            batch_user_emb = user_emb[batch_user_indices]

            # Compute scores: (batch_size, num_items)
            # batch_scores = model.predict(batch_user_indices, item_indices=None).cpu().numpy()
            batch_scores = torch.matmul(batch_user_emb, item_emb.T).cpu().numpy()

        elif config.model_name == 'MultiVAE':
            train_matrix = csr_matrix(
                (dataset.train_df['rating'].values,
                 (dataset.train_df['user_encoded'].values, dataset.train_df['item_encoded'].values)),
                shape=(dataset.num_users, dataset.num_items),
                dtype=np.float32)

            device = next(model.parameters()).device
            user_input = torch.tensor(train_matrix[batch_user_ids].toarray(),
                                      device=device, dtype=torch.float32)

            recon, _, _ = model(user_input)
            batch_scores = recon.cpu().numpy()

        elif config.model_name == 'ItemKNN':
            batch_scores = np.full((batch_size, dataset.num_items), float('-inf'))
            for i, user_id in enumerate(batch_user_ids):
                recommendations = model.predict(user_id, n_items=dataset.num_items)
                if len(recommendations) > 0 and len(recommendations[0]) > 0:
                    for item_id, score in recommendations[0]:
                        if 0 <= item_id < dataset.num_items:
                            batch_scores[i, int(item_id)] = score

        else:
            raise ValueError(f"Unsupported model type: {config.model_name}")

    # Mask training items efficiently
    batch_scores = mask_training_items_sparse(batch_scores, batch_user_ids, train_user_items)

    return batch_scores


def mask_training_items_sparse(scores, batch_user_ids, train_user_items):
    """
    Mask training items using sparse operations to avoid dense mask creation.
    """
    for i, user_id in enumerate(batch_user_ids):
        train_items = train_user_items.get(user_id, [])
        if train_items:
            # Only modify the specific indices, no dense matrix creation
            scores[i, train_items] = float('-inf')

    return scores


def process_batch_metrics(batch_scores, batch_relevance, batch_user_ids, user_test_items,
                          k_values, max_k, results, encoded_to_genres, dataset, config):
    """
    Process metrics for a batch of users using sparse operations.
    """
    batch_size = len(batch_user_ids)

    # Get top-k items for this batch
    sorted_indices = np.argsort(-batch_scores, axis=1)
    top_max_k_items = sorted_indices[:, :max_k]

    for k_val in k_values:
        top_k_items = top_max_k_items[:, :k_val]

        # Calculate standard metrics using sparse relevance
        batch_metrics = calculate_batch_standard_metrics_sparse(
            top_k_items, sorted_indices, batch_relevance, batch_scores, k_val
        )

        # Aggregate metrics
        results[k_val]['ndcg_scores'].extend(batch_metrics['ndcg'])
        results[k_val]['recall_scores'].extend(batch_metrics['recall'])
        results[k_val]['precision_scores'].extend(batch_metrics['precision'])
        results[k_val]['mrr_scores'].extend(batch_metrics['mrr'])
        results[k_val]['hit_rate_scores'].extend(batch_metrics['hit_rate'])

        # Process additional metrics efficiently
        for i, user_id in enumerate(batch_user_ids):
            user_top_k = top_k_items[i]

            # Update global tracking
            results[k_val]['all_recommended_items'].update(user_top_k)
            results[k_val]['all_global_recommendations'].extend(user_top_k)

            for item in user_top_k:
                results[k_val]['item_recommendation_freq'][item] += 1

            # Calculate additional metrics per user
            additional_metrics = calculate_user_additional_metrics_efficient(
                user_top_k, encoded_to_genres, dataset
            )

            results[k_val]['simpson_scores'].append(additional_metrics['simpson_index'])
            results[k_val]['intra_list_diversity_scores'].append(additional_metrics['intra_list_diversity'])
            results[k_val]['popularity_lift_scores'].append(additional_metrics['popularity_lift'])
            results[k_val]['avg_rec_popularity_scores'].append(additional_metrics['avg_rec_popularity'])
            results[k_val]['normalized_genre_entropy_scores'].append(additional_metrics['normalized_genre_entropy'])
            results[k_val]['unique_genres_count_scores'].append(additional_metrics['unique_genres_count'])

            # Community bias calculation
            if hasattr(config, 'item_labels_matrix_mask'):
                user_community_dist = calculate_user_community_distribution_efficient(
                    user_top_k, config.item_labels_matrix_mask, dataset.device
                )
                results[k_val]['user_community_distributions'].append(user_community_dist)


def calculate_batch_standard_metrics_sparse(top_k_items, sorted_indices, batch_relevance, batch_scores, k):
    """
    Calculate standard metrics for a batch using sparse relevance matrix.
    """
    batch_size = top_k_items.shape[0]

    batch_relevance_dense = batch_relevance.toarray().astype(bool)

    top_k_binary = np.zeros_like(batch_relevance_dense)
    user_indices = np.arange(batch_size)[:, np.newaxis]
    top_k_binary[user_indices, top_k_items] = 1

    # Calculate metrics vectorized for this batch
    intersection = np.sum(top_k_binary & batch_relevance_dense, axis=1)
    num_relevant_per_user = np.sum(batch_relevance_dense, axis=1)

    precision_scores = intersection / k

    recall_scores = np.where(num_relevant_per_user > 0, intersection / num_relevant_per_user, 0.0)

    hit_rate_scores = (intersection > 0).astype(float)

    ndcg_scores = calculate_ndcg_batch_sparse(top_k_items, batch_relevance_dense, k)

    mrr_scores = calculate_mrr_batch_sparse(sorted_indices, batch_relevance_dense)

    return {
        'ndcg': ndcg_scores,
        'recall': recall_scores,
        'precision': precision_scores,
        'mrr': mrr_scores,
        'hit_rate': hit_rate_scores
    }


def calculate_ndcg_batch_sparse(top_k_items, batch_relevance_dense, k):
    """
    Calculate NDCG@k for a batch using efficient operations.
    """
    batch_size = top_k_items.shape[0]
    user_indices = np.arange(batch_size)[:, np.newaxis]

    # Get relevance for top-k items
    top_k_relevance = batch_relevance_dense[user_indices, top_k_items]

    # Calculate DCG
    positions = np.arange(1, k + 1)
    discounts = 1.0 / np.log2(positions + 1)
    gains = 2 ** top_k_relevance.astype(float) - 1
    dcg = np.sum(gains * discounts, axis=1)

    # Calculate IDCG
    num_relevant_per_user = np.sum(batch_relevance_dense, axis=1)
    ideal_relevance = np.zeros((batch_size, k))

    for i in range(batch_size):
        num_rel = min(int(num_relevant_per_user[i]), k)
        if num_rel > 0:
            ideal_relevance[i, :num_rel] = 1

    ideal_gains = 2 ** ideal_relevance - 1
    idcg = np.sum(ideal_gains * discounts, axis=1)

    # NDCG = DCG / IDCG
    ndcg_scores = np.where(idcg > 0, dcg / idcg, 0.0)
    return ndcg_scores


def calculate_mrr_batch_sparse(sorted_indices, batch_relevance_dense):
    """
    Calculate MRR for a batch efficiently.
    """
    batch_size, num_items = batch_relevance_dense.shape
    mrr_scores = np.zeros(batch_size)

    for i in range(batch_size):
        user_relevance = batch_relevance_dense[i]
        if np.any(user_relevance):
            # Find positions of relevant items in the ranking
            for pos, item_idx in enumerate(sorted_indices[i]):
                if user_relevance[item_idx]:
                    mrr_scores[i] = 1 / (pos + 1)  # 1-based position
                    break

    return mrr_scores


def calculate_user_additional_metrics_efficient(top_k_items, encoded_to_genres, dataset):
    """
    Calculate additional metrics for a single user efficiently.
    """
    metrics = {}

    # Genre-based metrics
    if encoded_to_genres:
        # Get genres for recommended items
        item_genres = []
        for item_id in top_k_items:
            if item_id in encoded_to_genres:
                item_genres.extend(encoded_to_genres[item_id])

        if item_genres:
            # Simpson index
            genre_counts = {}
            for genre in item_genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

            total = sum(genre_counts.values())
            simpson_index = 1.0 - sum((count / total) ** 2 for count in genre_counts.values())
            metrics['simpson_index'] = simpson_index

            # Intra-list diversity for genres
            unique_genres = len(set(item_genres))
            metrics['intra_list_diversity'] = calculate_intra_list_diversity_genre(top_k_items, encoded_to_genres)

            # Genre entropy
            if len(genre_counts) > 1:
                entropy = 0.0
                for count in genre_counts.values():
                    p = count / total
                    entropy -= p * np.log2(p)
                max_entropy = np.log2(len(genre_counts))
                metrics['normalized_genre_entropy'] = entropy / max_entropy
            else:
                metrics['normalized_genre_entropy'] = 0.0

            metrics['unique_genres_count'] = unique_genres
        else:
            metrics.update({
                'simpson_index': 0.0,
                'intra_list_diversity': 0.0,
                'normalized_genre_entropy': 0.0,
                'unique_genres_count': 0
            })
    else:
        metrics.update({
            'simpson_index': 0.0,
            'intra_list_diversity': 0.0,
            'normalized_genre_entropy': 0.0,
            'unique_genres_count': 0
        })

    # Popularity metrics
    if len(top_k_items) > 0 and hasattr(dataset, 'item_popularities'):
        item_popularities = [dataset.item_popularities[item_id] for item_id in top_k_items
                             if item_id < len(dataset.item_popularities)]

        if item_popularities:
            avg_pop = np.mean(item_popularities)
            baseline_pop = np.mean(dataset.item_popularities.array)
            pop_lift = avg_pop / baseline_pop if baseline_pop > 0 else 1.0

            metrics['avg_rec_popularity'] = avg_pop
            metrics['popularity_lift'] = pop_lift
        else:
            metrics['avg_rec_popularity'] = 0.0
            metrics['popularity_lift'] = 1.0
    else:
        metrics['avg_rec_popularity'] = 0.0
        metrics['popularity_lift'] = 1.0

    return metrics


def calculate_user_community_distribution_efficient(recommended_items, item_labels_matrix_mask, device):
    """
    Calculate community distribution for recommended items, weighted by item's community count.

    Args:
        recommended_items: numpy array of recommended item IDs
        item_labels_matrix_mask: boolean matrix (nr_items, nr_communities)
                                where True means item i belongs to community j

    Returns:
        numpy array: distribution over communities
    """
    # Get mask for recommended items only
    rec_mask = item_labels_matrix_mask[recommended_items]

    # Count communities per item (number of True values per row)
    communities_per_item = rec_mask.sum(axis=1)

    # Weight each community membership by 1/communities_per_item
    # Use broadcasting to divide each row by its community count
    weights = 1.0 / communities_per_item[:, np.newaxis]
    weighted_mask = rec_mask * weights

    # Sum across all items to get final distribution
    community_sum = weighted_mask.sum(axis=0)
    community_distribution = community_sum / community_sum.sum() if community_sum.sum() > 0 else np.zeros(
        item_labels_matrix_mask.shape[1])

    return torch.tensor(community_distribution, device=device, dtype=torch.float32)


def aggregate_batch_results(results, k_values, dataset):
    """
    Aggregate results from all batches into final metrics.
    """
    final_results = {}

    for k_val in k_values:
        metrics = results[k_val]

        # Calculate aggregated metrics
        item_coverage = len(metrics['all_recommended_items']) / dataset.num_items

        # Gini coefficient
        if metrics['item_recommendation_freq']:
            recommendation_counts = np.array(list(metrics['item_recommendation_freq'].values()))
            recommendation_probs = recommendation_counts / recommendation_counts.sum()
            gini_index = calculate_gini_coefficient(recommendation_probs)
        else:
            gini_index = 0.0

        # Popularity calibration
        pop_miscalibration = popularity_miscalibration(
            metrics['all_global_recommendations'], dataset
        )

        # Community bias
        user_community_bias = 0.0
        if metrics['user_community_distributions']:
            all_user_community_dists = torch.stack(metrics['user_community_distributions'])
            user_bias, _ = get_community_bias(item_communities_each_user_dist=all_user_community_dists)
            user_community_bias = float(torch.mean(user_bias)) if user_bias is not None else 0.0

        final_results[k_val] = {
            'ndcg': np.mean(metrics['ndcg_scores']) if metrics['ndcg_scores'] else 0.0,
            'recall': np.mean(metrics['recall_scores']) if metrics['recall_scores'] else 0.0,
            'precision': np.mean(metrics['precision_scores']) if metrics['precision_scores'] else 0.0,
            'mrr': np.mean(metrics['mrr_scores']) if metrics['mrr_scores'] else 0.0,
            'hit_rate': np.mean(metrics['hit_rate_scores']) if metrics['hit_rate_scores'] else 0.0,
            'item_coverage': item_coverage,
            'gini_index': gini_index,
            'simpson_index_genre': np.mean(metrics['simpson_scores']) if metrics['simpson_scores'] else 0.0,
            'intra_list_diversity': np.mean(metrics['intra_list_diversity_scores']) if metrics[
                'intra_list_diversity_scores'] else 0.0,
            'popularity_lift': np.mean(metrics['popularity_lift_scores']) if metrics['popularity_lift_scores'] else 0.0,
            'avg_rec_popularity': np.mean(metrics['avg_rec_popularity_scores']) if metrics[
                'avg_rec_popularity_scores'] else 0.0,
            'normalized_genre_entropy': np.mean(metrics['normalized_genre_entropy_scores']) if metrics[
                'normalized_genre_entropy_scores'] else 0.0,
            'unique_genres_count': np.mean(metrics['unique_genres_count_scores']) if metrics[
                'unique_genres_count_scores'] else 0.0,
            'pop_miscalibration': pop_miscalibration,
            'user_community_bias': user_community_bias
        }

    return final_results


def popularity_miscalibration(all_recommended_items, dataset):
    """
    Calculate popularity calibration - measures how well the recommendation popularity
    distribution matches the catalog popularity distribution using KL divergence.

    Lower values indicate better calibration (closer to catalog distribution).

    Args:
        all_recommended_items: list of all recommended items across all users
        dataset: dataset object with item_popularities
        num_bins: number of bins for histogram comparison

    Returns:
        float: KL divergence between recommendation and catalog distributions
    """
    if not all_recommended_items:
        return float('inf')

    catalog_popularities = np.array(dataset.item_popularities.array)

    # Get popularities for recommended items
    recommended_popularities = []
    for item_id in all_recommended_items:
        if 0 <= item_id < len(catalog_popularities):
            recommended_popularities.append(catalog_popularities[item_id])

    if not recommended_popularities:
        return float('inf')

    recommended_popularities = np.array(recommended_popularities)

    # Create bins based on catalog popularity range
    min_pop, max_pop = np.min(catalog_popularities), np.max(catalog_popularities)
    if min_pop == max_pop:
        return 0.0  # Perfect calibration if all items have same popularity

    # Use equal-width bins across the popularity range
    num_bins = len(dataset.val_df)  # Use number of validation items as number of bins
    bins = np.linspace(min_pop, max_pop, num_bins + 1)

    # Calculate normalized histograms (probability distributions)
    catalog_hist, _ = np.histogram(catalog_popularities, bins=bins)
    recommended_hist, _ = np.histogram(recommended_popularities, bins=bins)

    # Convert to probability distributions
    catalog_dist = catalog_hist / np.sum(catalog_hist) if np.sum(catalog_hist) > 0 else np.ones(num_bins) / num_bins
    recommended_dist = recommended_hist / np.sum(recommended_hist) if np.sum(recommended_hist) > 0 else np.ones(
        num_bins) / num_bins

    # Add small epsilon to avoid log(0) in KL divergence
    epsilon = 1e-10
    catalog_dist = np.maximum(catalog_dist, epsilon)
    recommended_dist = np.maximum(recommended_dist, epsilon)

    # Calculate KL divergence: KL(P||Q) = sum(P(x) * log(P(x)/Q(x)))
    # where P is recommendation distribution, Q is catalog distribution
    kl_divergence = np.sum(recommended_dist * np.log(recommended_dist / catalog_dist))

    return kl_divergence


def evaluate_current_model_ndcg_vectorized(model, dataset, model_type='LightGCN', k=10):
    """
    Memory-efficient NDCG evaluation for early stopping using batching.
    """
    if model_type != 'ItemKNN':
        model.eval()

    train_user_items = dataset.train_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()
    user_val_items = dataset.val_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

    valid_user_ids = np.array([uid for uid in user_val_items.keys() if uid < dataset.num_users])

    # Create sparse relevance matrix
    relevance_sparse = create_sparse_relevance_matrix(valid_user_ids, user_val_items, dataset.num_items)

    # Process in batches
    batch_size = min(1024, len(valid_user_ids))
    all_ndcg_scores = []

    config_mock = type('Config', (), {'model_name': model_type})()

    for batch_start in range(0, len(valid_user_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_user_ids))
        batch_user_ids = valid_user_ids[batch_start:batch_end]

        # Get scores for this batch
        batch_scores = get_batch_user_scores(model, dataset, config_mock, batch_user_ids, train_user_items)

        # Get top-k items
        top_k_items = np.argsort(-batch_scores, axis=1)[:, :k]

        # Get relevance for this batch
        batch_relevance = relevance_sparse[batch_start:batch_end].toarray().astype(bool)

        # Calculate NDCG for this batch
        batch_ndcg = calculate_ndcg_batch_sparse(top_k_items, batch_relevance, k)
        all_ndcg_scores.extend(batch_ndcg)

    return np.mean(all_ndcg_scores)


