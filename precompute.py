import numpy as np
import torch
import torch_geometric
import scipy.sparse as sp
from sknetwork.clustering import Leiden
import os
from scipy.stats import binom


def get_community_labels(config, adj_np, save_path='dataset/ml-100k/saved', get_probs=True, force_bipartite=True):
    # read and return them if already computed locally
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if f'item_labels_matrix.csv' in os.listdir(save_path):
        return torch.tensor(
            np.loadtxt(f'{save_path}/user_labels_matrix.csv', delimiter=','),
            dtype=torch.int64,
            device=device
        ), torch.tensor(
            np.loadtxt(f'{save_path}/item_labels_matrix.csv', delimiter=','),
            dtype=torch.int64,
            device=device
        )

    adj_csr = sp.csr_matrix((adj_np[:, 2].astype(float), (adj_np[:, 0].astype(int), adj_np[:, 1].astype(int))))

    if config.dataset_name == 'ml-100k':
        resolution = 1.635
    elif config.dataset_name == 'ml-1m':
        resolution = 1.5625
    else:  # config.dataset_name == 'lastfm':
        resolution = 1.71

    # for i in [1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.56, 1.5625, 1.565, 1.5675, 1.57, 1.58, 1.59, 1.6, 1.62, 1.635, 1.65, 1.7, 1.71, 1.72, 1.73, 1.74, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0]:
    #     detect_obj = Leiden(resolution=i, return_aggregate=False, sort_clusters=True)
    #     detect_obj.fit(adj_csr, force_bipartite=force_bipartite)
    #     info = f'Resolution: {i}, Number of user clusters: {detect_obj.probs_row_.shape[1]}, Number of item clusters: {detect_obj.probs_col_.shape[1]}, Modularity float: {detect_obj.log}'
    #     print(info)

    detect_obj = Leiden(resolution=resolution, return_aggregate=False, sort_clusters=True)
    detect_obj.fit(adj_csr, force_bipartite=force_bipartite)

    info = f'Resolution: {resolution}, Number of user clusters: {detect_obj.probs_row_.shape[1]}, Number of item clusters: {detect_obj.probs_col_.shape[1]}, Modularity float: {detect_obj.log}'
    print(info)
    with open(f'{save_path}/leiden_fit_information.txt', 'w') as f:
        f.write(info)

    user_probs = detect_obj.probs_row_.toarray()
    item_probs = detect_obj.probs_col_.toarray()
    # argsort the rel_rec_freqs to get the highest probability label
    # only take columns that have not all zeros
    user_probs = user_probs[:, np.any(user_probs, axis=0)]
    item_probs = item_probs[:, np.any(item_probs, axis=0)]

    user_probs_argsorted = np.argsort(user_probs, axis=1)[:, ::-1]
    item_probs_argsorted = np.argsort(item_probs, axis=1)[:, ::-1]

    # the detect_obj.labels doesn't always assign the most probable label for some reason so we have to do
    user_labels = user_probs_argsorted[:, 0]
    item_labels = item_probs_argsorted[:, 0]

    # keeping for community connectivity matrix
    np.savetxt(f'{save_path}/user_labels_raw.csv', user_labels, delimiter=",", fmt='% 4d')
    np.savetxt(f'{save_path}/item_labels_raw.csv', item_labels, delimiter=",", fmt='% 4d')

    user_probs_sorted = np.sort(user_probs, axis=1)[:, ::-1]
    item_probs_sorted = np.sort(item_probs, axis=1)[:, ::-1]

    user_community_thresholds = binomial_significance_threshold(n_interactions=config.user_degrees,
                                                                  n_categories=user_probs.shape[1],
                                                                  alpha=0.05)
    item_community_thresholds = binomial_significance_threshold(n_interactions=config.item_degrees,
                                                                  n_categories=item_probs.shape[1],
                                                                  alpha=0.05)

    # Replace -1 with another value (e.g., 0 or a special flag like -99)
    # user_labels = np.where(user_probs_sorted[:, 0] < user_community_thresholds, -1, user_labels)
    # item_labels = np.where(item_probs_sorted[:, 0] < item_community_thresholds, -1, item_labels)

    user_labels_sorted_matrix = np.full((user_labels.shape[0], user_probs.shape[1]), fill_value=-1, dtype=np.int64)
    item_labels_sorted_matrix = np.full((item_labels.shape[0], item_probs.shape[1]), fill_value=-1, dtype=np.int64)

    user_labels_matrix_mask = np.zeros(user_labels_sorted_matrix.shape, dtype=np.int64)
    item_labels_matrix_mask = np.zeros(item_labels_sorted_matrix.shape, dtype=np.int64)

    for i in range(0, user_probs.shape[1]):
    # for i in range(1):  # we dont use multiple memberships anymore as soft clustering is bullshit
        user_labels_i = np.where(user_probs_sorted[:, i] > user_community_thresholds, user_probs_argsorted[:, i], -1)
        item_labels_i = np.where(item_probs_sorted[:, i] > item_community_thresholds, item_probs_argsorted[:, i], -1)
        ### testing only one label
        # user_labels_i = user_labels
        # item_labels_i = item_labels
        ###

        if np.all(user_labels_i == -1) and np.all(item_labels_i == -1):
            break
        user_labels_sorted_matrix[:, i] = user_labels_i
        item_labels_sorted_matrix[:, i] = item_labels_i

        user_rows_with_label = np.where(user_labels_sorted_matrix[:, i] != -1)[0]
        user_labels_matrix_mask[user_rows_with_label, user_labels_i[user_rows_with_label]] = 1
        item_rows_with_label = np.where(item_labels_sorted_matrix[:, i] != -1)[0]
        item_labels_matrix_mask[item_rows_with_label, item_labels_i[item_rows_with_label]] = 1

    # keep only columns that have not all -1s
    user_labels_sorted_matrix = user_labels_sorted_matrix[:, np.any(user_labels_sorted_matrix != -1, axis=0)]
    item_labels_sorted_matrix = item_labels_sorted_matrix[:, np.any(item_labels_sorted_matrix != -1, axis=0)]
    # as clusters are sorted by size, we can remove columns that have all zeros
    user_col_drop = np.any(user_labels_matrix_mask != 0, axis=0)
    item_col_drop = np.any(item_labels_matrix_mask != 0, axis=0)
    user_labels_matrix_mask = user_labels_matrix_mask[:, user_col_drop]
    item_labels_matrix_mask = item_labels_matrix_mask[:, item_col_drop]
    ### keeping only first label for each user/item
    user_labels_matrix_mask = np.zeros(user_labels_matrix_mask.shape, dtype=np.int64)
    item_labels_matrix_mask = np.zeros(item_labels_matrix_mask.shape, dtype=np.int64)
    # make only 1 at mask for highest probability label
    user_labels_matrix_mask[np.arange(user_labels_matrix_mask.shape[0]), user_labels_sorted_matrix[:, 0]] = 1
    item_labels_matrix_mask[np.arange(item_labels_matrix_mask.shape[0]), item_labels_sorted_matrix[:, 0]] = 1

    user_probs = user_probs[:, user_col_drop]
    item_probs = item_probs[:, item_col_drop]

    np.savetxt(f'{save_path}/user_labels_processed.csv', user_labels, delimiter=",", fmt='% 4d')
    np.savetxt(f'{save_path}/item_labels_processed.csv', item_labels, delimiter=",", fmt='% 4d')

    np.savetxt(f'{save_path}/user_labels_matrix.csv', user_labels_sorted_matrix, delimiter=",", fmt='% 4d')
    np.savetxt(f'{save_path}/item_labels_matrix.csv', item_labels_sorted_matrix, delimiter=",", fmt='% 4d')

    # for dropout
    np.savetxt(f'{save_path}/user_labels_matrix_mask.csv', user_labels_matrix_mask, delimiter=",", fmt='% 4d')
    np.savetxt(f'{save_path}/item_labels_matrix_mask.csv', item_labels_matrix_mask, delimiter=",", fmt='% 4d')

    if get_probs:
        np.savetxt(f'{save_path}/user_labels_probs.csv', user_probs,
                   delimiter=",", fmt='%.4f')
        np.savetxt(f'{save_path}/item_labels_probs.csv', item_probs,
                   delimiter=",", fmt='%.4f')

    return torch.tensor(user_labels_sorted_matrix, dtype=torch.int64, device=device), torch.tensor(item_labels_sorted_matrix, dtype=torch.int64, device=device)


def binomial_significance_threshold(n_interactions, n_categories=18, alpha=0.05):
    """
    Returns the smallest count T (clamped to n_interactions) such that
    P(X >= T) <= alpha/n_categories, and its proportion T/n_interactions <= 1.
    """
    p = 1.0 / n_categories
    alpha_per_test = alpha / n_categories

    n_arr = np.atleast_1d(n_interactions)
    # find k_values with P(X > k_values) ≤ alpha_per_test, so T=k_values+1 may exceed n -> clamp next
    raw_thresh = binom.isf(alpha_per_test, n_arr, p) + 1
    thresh = np.minimum(raw_thresh, n_arr)           # clamp to max trials
    props = thresh / n_arr                           # to normalize ≤ 1
    # props = thresh
    if np.isscalar(n_interactions):
        return float(props[0])
    return props


def get_power_users_items(adj_tens, user_com_labels, item_com_labels,
                          users_top_percent=0.01, items_top_percent=0.01,
                          save_path='/dataset/saved/ml-100k'):
    """
    Get the indices of the top users and items based on their degree
    :param adj_tens: torch.tensor, format (n, 3) with (user, item, rating)
    :param user_com_labels: torch.tensor format (n_users, m_labels (user with most labels)), community labels for each user
    :param item_com_labels: torch.tensor format (n_items, m_labels (item with most labels)), community labels for each item
    :param users_top_percent: float, percentage of top users to keep
    :param items_top_percent: float, percentage of top items to keep
    :param save_path: str, path to save the output files
    :return: torch.tensor, indices of top users, torch.tensor, indices of top items
    """
    # read and return them if already computed locally
    device = adj_tens.device  # Get the device of the input tensor
    if f'power_user_ids_top{users_top_percent}.csv' in os.listdir(
            save_path) and f'power_item_ids_top{items_top_percent}.csv' in os.listdir(save_path):
        return torch.tensor(
            np.loadtxt(f'{save_path}/power_user_ids_top{users_top_percent}.csv'),
            dtype=torch.int64,
            device=device
        ), torch.tensor(
            np.loadtxt(f'{save_path}/power_item_ids_top{items_top_percent}.csv'),
            dtype=torch.int64,
            device=device)

    power_user_ids = torch.tensor([], dtype=torch.int64, device=device)
    power_item_ids = torch.tensor([], dtype=torch.int64, device=device)
    user_com_labels = user_com_labels.to(device)
    item_com_labels = item_com_labels.to(device)

    if users_top_percent > 0:
        # Get significant communities (containing at least 1% of users/items)
        # users and items are assigned multiple communities if they exceed a confidence threshold
        unique_user_labels, user_count = torch.unique(torch.flatten(user_com_labels), return_counts=True)
        all_users = torch.unique(adj_tens[:, 0])
        top_user_com_idx = torch.where(user_count >= 0.01 * len(all_users))[0]

        for user_label in unique_user_labels[top_user_com_idx]:
            if user_label == -1:  # no interpretation; -1 is used to assign no community, over all columns
                continue
            # Get user ids/indices if the user label is one of the users' community labels (if user label is in the row)
            users_idx = torch.where((user_com_labels == user_label).any(dim=1))[0]
            # From these node ids, get the edges
            user_edges = adj_tens[torch.isin(adj_tens[:, 0], users_idx)]
            # Get highest degrees of users
            user_degrees = torch_geometric.utils.degree(user_edges[:, 0])
            sorted_users_idx = torch.argsort(user_degrees, descending=True)
            top_x_percent_user_idx = int(len(users_idx) * users_top_percent) + 1
            top_users = sorted_users_idx[:top_x_percent_user_idx].flatten()
            power_user_ids = torch.cat((power_user_ids, top_users))

    # Process items by community
    if items_top_percent > 0:
        # Get significant communities (containing at least 1% of users/items)
        # users and items are assigned multiple communities if they exceed a confidence threshold
        unique_item_labels, item_count = torch.unique(torch.flatten(item_com_labels), return_counts=True)
        all_items = torch.unique(adj_tens[:, 1])
        top_item_com_idx = torch.where(item_count >= 0.01 * len(all_items))[0]

        for item_label in unique_item_labels[top_item_com_idx]:
            if item_label == -1:  # no interpretation; -1 is used to assign no community, over all columns
                continue
            # Get items from this community
            items_idx = torch.where((item_com_labels == item_label).any(dim=1))[0]
            # From these node ids, get the edges
            item_edges = adj_tens[torch.isin(adj_tens[:, 1], items_idx)]
            # Get highest degrees of items
            item_degrees = torch_geometric.utils.degree(item_edges[:, 1])
            sorted_items_idx = torch.argsort(item_degrees, descending=True)
            top_x_percent_item_idx = int(len(items_idx) * items_top_percent) + 1
            top_items = sorted_items_idx[:top_x_percent_item_idx].flatten()
            power_item_ids = torch.cat((power_item_ids, top_items))

    power_user_ids = torch.unique(power_user_ids)
    power_item_ids = torch.unique(power_item_ids)

    if users_top_percent > 0:
        np.savetxt(f'{save_path}/power_user_ids_top{users_top_percent}.csv',
                   power_user_ids.cpu().numpy(), delimiter=",", fmt='% 4d')

    if items_top_percent > 0:
        np.savetxt(f'{save_path}/power_item_ids_top{items_top_percent}.csv',
                   power_item_ids.cpu().numpy(), delimiter=",", fmt='% 4d')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.tensor(power_user_ids, dtype=torch.int64, device=device), torch.tensor(power_item_ids, dtype=torch.int64, device=device)


def get_community_connectivity_matrix(adj_tens, user_com_labels, item_com_labels):
    """
    Get the connectivity matrix of the communities using vectorized operations on GPU.

    :param adj_tens: torch.tensor, adjacency matrix with format (n, 3) containing (user_id, item_id, rating)
    :param user_com_labels: torch.tensor, community labels for each user
    :param item_com_labels: torch.tensor, community labels for each item
    :return: torch.tensor, connectivity matrix
    """
    device = adj_tens.device  # Get the device of the input tensor

    # Extract user and item indices
    user_indices = adj_tens[:, 0].long()
    item_indices = adj_tens[:, 1].long()

    # Get community assignments for each interaction
    user_communities = user_com_labels[user_indices]
    item_communities = item_com_labels[item_indices]

    # Get number of communities
    n_user_communities = user_com_labels.max().item() + 1
    n_item_communities = item_com_labels.max().item() + 1

    # Initialize connectivity matrix
    connectivity_matrix = torch.zeros((n_user_communities, n_item_communities), device=device)

    # Create a tensor of ones to count each interaction
    ones = torch.ones(user_communities.size(0), device=device)

    # Use index_put_ with accumulate=True to count connections
    connectivity_matrix.index_put_(
        (user_communities, item_communities),
        ones,
        accumulate=True
    )

    return connectivity_matrix  # raw interaction counts, not normalized


def get_user_item_community_connectivity_matrices(adj_tens, user_com_labels, item_com_labels):
    """
    Measure community bias from this output!!

    Get tensors counting:
    1. For each user, the number of edges to each item community
    2. For each item, the number of edges to each user community

    Considers all community assignments for each user/item across all columns of label matrices.

    Args:
        adj_tens: Tensor of shape (nr_edges, 3) containing [user_id, item_id, strength]
        user_com_labels: Tensor containing user community labels (users x communities)
        item_com_labels: Tensor containing item community labels (items x communities)

    Returns:
        item_communities_each_user: Tensor of shape (nr_users, nr_item_communities)
        user_communities_each_item: Tensor of shape (nr_items, nr_user_communities)
    """
    device = adj_tens.device

    # Extract users and items from adjacency tensor
    users = adj_tens[:, 0].long()
    items = adj_tens[:, 1].long()

    # Get dimensions
    nr_users = max(torch.max(users).item() + 1, user_com_labels.size(0))
    nr_items = max(torch.max(items).item() + 1, item_com_labels.size(0))
    nr_user_communities = torch.max(user_com_labels).item() + 1
    nr_item_communities = torch.max(item_com_labels).item() + 1

    # Initialize accumulator sparse tensors
    item_communities_each_user = torch.zeros((nr_users, nr_item_communities), device=device)
    user_communities_each_item = torch.zeros((nr_items, nr_user_communities), device=device)

    # Get number of columns in community label matrices
    ### testing one label with unsqueeze()
    # user_com_labels.unsqueeze(dim=1)
    # item_com_labels.unsqueeze(dim=1)
    user_com_cols = user_com_labels.size(1)
    item_com_cols = item_com_labels.size(1)
    ###

    # Process each user community column
    for col in range(user_com_cols):
        user_communities = user_com_labels[users, col]
        # user_communities = user_com_labels[users]
        valid_mask = user_communities >= 0  # Skip -1 values (no community assignment)

        if not valid_mask.any():
            continue

        valid_users = users[valid_mask]
        valid_items = items[valid_mask]
        valid_communities = user_communities[valid_mask]

        # Create sparse indices
        item_user_comm_indices = torch.stack([valid_items, valid_communities])

        # Create values (all 1s to count occurrences)
        values = torch.ones_like(valid_users, dtype=torch.float, device=device)

        # Accumulate counts
        temp_tensor = torch.sparse.FloatTensor(
            item_user_comm_indices,
            values,
            torch.Size([nr_items, nr_user_communities])
        ).to_dense()

        user_communities_each_item += temp_tensor

    # Process each item community column
    for col in range(item_com_cols):
        item_communities = item_com_labels[items, col]
        # item_communities = item_com_labels[items]
        valid_mask = item_communities >= 0  # Skip -1 values (no community assignment)

        if not valid_mask.any():
            continue

        valid_users = users[valid_mask]
        valid_items = items[valid_mask]
        valid_communities = item_communities[valid_mask]

        # Create sparse indices
        user_item_comm_indices = torch.stack([valid_users, valid_communities])

        # Create values (all 1s to count occurrences)
        values = torch.ones_like(valid_items, dtype=torch.float, device=device)

        # Accumulate counts
        temp_tensor = torch.sparse.FloatTensor(
            user_item_comm_indices,
            values,
            torch.Size([nr_users, nr_item_communities])
        ).to_dense()

        item_communities_each_user += temp_tensor

    return torch.tensor(item_communities_each_user, dtype=torch.int32), torch.tensor(user_communities_each_item, dtype=torch.int32)


def get_biased_edges_mask(config, adj_tens, user_com_labels_mask, item_com_labels_mask,
                          user_community_connectivity_matrix_distribution,
                          item_community_connectivity_matrix_distribution,):

    user_community_thresholds = binomial_significance_threshold(n_interactions=config.user_degrees,
                                                                  n_categories=user_community_connectivity_matrix_distribution.shape[1],
                                                                  alpha=0.05)
    item_community_thresholds = binomial_significance_threshold(n_interactions=config.item_degrees,
                                                                  n_categories=item_community_connectivity_matrix_distribution.shape[1],
                                                                  alpha=0.05)

    # get masks for communities that are row wise above the threshold
    users_com_connectivity_mask = user_community_connectivity_matrix_distribution > torch.tensor(user_community_thresholds, device=config.device).unsqueeze(1)
    items_com_connectivity_mask = item_community_connectivity_matrix_distribution > torch.tensor(item_community_thresholds, device=config.device).unsqueeze(1)

    ### uncomment to use only the highest community connectivity for each user/item
    # highest_row_indices = torch.argmax(user_community_connectivity_matrix_distribution, dim=1)
    # highest_item_row_indices = torch.argmax(item_community_connectivity_matrix_distribution, dim=1)
    # users_com_connectivity_mask = torch.zeros(user_community_connectivity_matrix_distribution.shape, dtype=torch.bool, device=config.device)
    # items_com_connectivity_mask = torch.zeros(item_community_connectivity_matrix_distribution.shape, dtype=torch.bool, device=config.device)
    # users_com_connectivity_mask[torch.arange(user_community_connectivity_matrix_distribution.shape[0]), highest_row_indices] = 1
    # items_com_connectivity_mask[torch.arange(item_community_connectivity_matrix_distribution.shape[0]), highest_item_row_indices] = 1

    # get indices where in user/item com masks at least one community is above threshold
    users_com_labels_mask_rows = torch.any(user_com_labels_mask > 0, dim=1)
    items_com_labels_mask_rows = torch.any(item_com_labels_mask > 0, dim=1)

    user_label_mask = users_com_labels_mask_rows[adj_tens[:, 0]]
    idx_item_label_mask = items_com_labels_mask_rows[adj_tens[:, 1]]

    idx_biased_user_nodes = adj_tens[user_label_mask][:, 0]
    idx_biased_user_nodes_items = adj_tens[user_label_mask][:, 1]
    idx_biased_item_nodes = adj_tens[idx_item_label_mask][:, 1]
    idx_biased_item_nodes_users = adj_tens[idx_item_label_mask][:, 0]

    biased_user_nodes_item_com_con = users_com_connectivity_mask[idx_biased_user_nodes]
    biased_item_nodes_user_com_con = items_com_connectivity_mask[idx_biased_item_nodes]

    biased_item_nodes_communities = item_com_labels_mask[idx_biased_user_nodes_items].bool()
    biased_user_nodes_communities = user_com_labels_mask[idx_biased_item_nodes_users].bool()

    user_com_mask_nonzero_idx = user_label_mask.nonzero(as_tuple=True)[0]
    item_com_mask_nonzero_idx = idx_item_label_mask.nonzero(as_tuple=True)[0]

    true_user_indices = torch.any(biased_user_nodes_item_com_con & biased_item_nodes_communities, dim=1)
    true_item_indices = torch.any(biased_item_nodes_user_com_con & biased_user_nodes_communities, dim=1)

    biased_user_edges_mask = torch.zeros(adj_tens.shape[0], dtype=torch.bool, device=adj_tens.device)
    biased_item_edges_mask = torch.zeros(adj_tens.shape[0], dtype=torch.bool, device=adj_tens.device)
    biased_user_edges_mask[user_com_mask_nonzero_idx[true_user_indices]] = True
    biased_item_edges_mask[item_com_mask_nonzero_idx[true_item_indices]] = True

    return biased_user_edges_mask, biased_item_edges_mask


