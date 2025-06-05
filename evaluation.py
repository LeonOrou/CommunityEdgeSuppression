import numpy as np
import torch
import json
from collections import defaultdict, Counter


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

    encoded_to_genres = load_genre_mapping(dataset)

    if stage == 'full_train':
        dataset.val_df = dataset.test_df

    # Create a set of training interactions for each user to exclude from evaluation
    # train_user_items = defaultdict(set)
    # for _, row in dataset.train_df.iterrows():
    #     train_user_items[row['user_encoded']].add(row['item_encoded'])
    # Using pandas groupby (more efficient):
    train_user_items = dataset.train_df.groupby('user_encoded')['item_encoded'].apply(set).to_dict()

    # Initialize metric storage for each k_values (MODIFY THIS)
    metrics_by_k = {k_val: {
        'ndcg_scores': [],
        'recall_scores': [],
        'precision_scores': [],
        'mrr_scores': [],
        'hit_rate_scores': [],
        'all_recommended_items': set(),
        # ADD THESE NEW METRICS
        'simpson_scores': [],
        'intra_list_diversity_scores': [],
        'popularity_lift_scores': [],
        'avg_recommended_popularity_scores': [],
        'normalized_genre_entropy_scores': [],
        'unique_genres_count_scores': []
    } for k_val in k_values}

    # Store recommendation frequencies and global recommendations for calibration
    item_recommendation_freq_by_k = {k_val: defaultdict(int) for k_val in k_values}
    all_global_recommendations_by_k = {k_val: [] for k_val in k_values}  # ADD THIS

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
            top_max_k_items = np.argsort(scores_filtered)[::-1][:max_k]

            # Get full ranking for MRR calculation
            full_ranking = np.argsort(scores_filtered)[::-1]

            # Calculate metrics for each k_values value
            for k_val in k_values:
                # Get top-k_values items for this k_values value
                top_k_items = top_max_k_items[:k_val]

                # Add to recommended items set for coverage calculation
                metrics_by_k[k_val]['all_recommended_items'].update(top_k_items)

                # Add to global recommendations for calibration (ADD THIS)
                all_global_recommendations_by_k[k_val].extend(top_k_items)

                # Update recommendation frequency
                for item in top_k_items:
                    item_recommendation_freq_by_k[k_val][item] += 1

                # Calculate existing metrics
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

                # ADD NEW METRICS CALCULATION
                additional_metrics = calculate_additional_metrics(
                    top_k_items, encoded_to_genres, dataset
                )

                metrics_by_k[k_val]['simpson_scores'].append(additional_metrics['simpson_index'])
                metrics_by_k[k_val]['intra_list_diversity_scores'].append(additional_metrics['intra_list_diversity'])
                metrics_by_k[k_val]['popularity_lift_scores'].append(additional_metrics['popularity_lift'])
                metrics_by_k[k_val]['avg_recommended_popularity_scores'].append(
                    additional_metrics['avg_recommended_popularity'])
                metrics_by_k[k_val]['normalized_genre_entropy_scores'].append(
                    additional_metrics['normalized_genre_entropy'])
                metrics_by_k[k_val]['unique_genres_count_scores'].append(additional_metrics['unique_genres_count'])

    # Compute final metrics for each k_values (MODIFY THIS SECTION)
    results = {}
    for k_val in k_values:
        # Calculate averages for existing metrics
        ndcg_scores = metrics_by_k[k_val]['ndcg_scores']
        recall_scores = metrics_by_k[k_val]['recall_scores']
        precision_scores = metrics_by_k[k_val]['precision_scores']
        mrr_scores = metrics_by_k[k_val]['mrr_scores']
        hit_rate_scores = metrics_by_k[k_val]['hit_rate_scores']

        # Calculate averages for new metrics
        simpson_scores = metrics_by_k[k_val]['simpson_scores']
        intra_list_diversity_scores = metrics_by_k[k_val]['intra_list_diversity_scores']
        popularity_lift_scores = metrics_by_k[k_val]['popularity_lift_scores']
        avg_recommended_popularity_scores = metrics_by_k[k_val]['avg_recommended_popularity_scores']
        normalized_genre_entropy_scores = metrics_by_k[k_val]['normalized_genre_entropy_scores']
        unique_genres_count_scores = metrics_by_k[k_val]['unique_genres_count_scores']

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

        # Calculate popularity calibration
        popularity_calibration = calculate_popularity_calibration(
            all_global_recommendations_by_k[k_val], dataset
        )

        results[k_val] = {
            # Existing metrics
            'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'recall': np.mean(recall_scores) if recall_scores else 0.0,
            'precision': np.mean(precision_scores) if precision_scores else 0.0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'hit_rate': np.mean(hit_rate_scores) if hit_rate_scores else 0.0,
            'item_coverage': item_coverage,
            'gini_index': gini_index,
            'simpson_index': 1.0 - gini_index,  # Keep your existing calculation

            # New metrics
            'simpson_index_genre': np.mean(simpson_scores) if simpson_scores else 0.0,
            'intra_list_diversity': np.mean(intra_list_diversity_scores) if intra_list_diversity_scores else 0.0,
            'popularity_lift': np.mean(popularity_lift_scores) if popularity_lift_scores else 0.0,
            'avg_recommended_popularity': np.mean(
                avg_recommended_popularity_scores) if avg_recommended_popularity_scores else 0.0,
            'normalized_genre_entropy': np.mean(
                normalized_genre_entropy_scores) if normalized_genre_entropy_scores else 0.0,
            'unique_genres_count': np.mean(unique_genres_count_scores) if unique_genres_count_scores else 0.0,
            'popularity_calibration': popularity_calibration
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
    # Using pandas groupby (more efficient):
    train_user_items = dataset.train_df.groupby('user_encoded')['item_encoded'].apply(set).to_dict()

    with torch.no_grad():
        # Use the COMPLETE graph (including test edges) for embeddings
        user_emb, item_emb = model(dataset.complete_edge_index, dataset.complete_edge_weight)

        # Group validation interactions by user
        user_val_items = dataset.val_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

        for user_id, true_items in user_val_items.items():
            if user_id >= dataset.num_users:
                continue

            # Get user embedding and compute scores for all items
            user_embedding = user_emb[user_id:user_id + 1]
            scores = torch.matmul(user_embedding, item_emb.T).squeeze().cpu().numpy()

            # Exclude items that the user interacted with during training
            train_items = list(train_user_items[user_id])
            scores_filtered = scores.copy()
            scores_filtered[train_items] = float('-inf')

            # Calculate NDCG
            ndcg = calculate_ndcg(true_items, scores_filtered, k)
            ndcg_scores.append(ndcg)

    # Return average NDCG
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def load_genre_mapping(dataset):
    """
    Load genre labels and create mapping from encoded item IDs to genres
    """
    # Load genre labels from JSON file
    with open(f'dataset/{dataset.name}/saved/item_genre_labels_{dataset.name}.json', 'r') as f:
        item_genre_labels = json.load(f)

    # Create mapping from encoded item ID to genres
    # Need to map: original_item_id -> encoded_item_id -> genres
    # item_id_to_encoded = {}
    # for _, row in dataset.complete_df.iterrows():
    #     item_id_to_encoded[str(row['item_id'])] = row['item_encoded']
    item_id_to_encoded = dataset.complete_df.set_index('item_id')['item_encoded'].to_dict()
    # Convert keys to strings for json indexing
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