import pickle
import numpy as np
import torch
import torch_geometric
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sknetwork.clustering import Leiden, Louvain
from argparse import ArgumentParser
import os


def get_community_labels(config, adj_np, algorithm='Leiden', save_path='dataset/ml-100k', get_probs=True, force_bipartite=True):
    ### TODO: make it work with Louvain algorithm

    # read and return them if already computed locally
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if f'user_labels_Leiden_processed.csv' in os.listdir(f'dataset/{config.dataset}'):
        return torch.tensor(
            np.loadtxt(f'dataset/{config.dataset}/user_labels_Leiden_matrix.csv', dtype=np.int32),
            dtype=torch.int32,
            device=device
        ), torch.tensor(
            np.loadtxt(f'dataset/{config.dataset}/item_labels_Leiden_matrix.csv', dtype=np.int32),
            dtype=torch.int32,
            device=device
        )


    # check if column 0 and 1 do not intersect, no ambiguity (often the ids are two sets so they both start at 1)
    # Ids start at 1 in MovieLens dataset
    max_userId = np.max(adj_np[:, 0])
    min_userId = np.min(adj_np[:, 0])
    max_itemId = np.max(adj_np[:, 1])
    min_itemId = np.min(adj_np[:, 1])
    # if not np.all(adj_np[:, 0] != adj_np[:, 1]):
    #     adj_np[:, 1] = adj_np[:,
    #                    1] + max_userId + min_userId  # min_userId is 1, if its 0: no gap between new user and item ids

    # make undirected such that also items get community labels
    # adj_np = np.concatenate([adj_np, adj_np[:, [1, 0, 2]]])
    adj_csr = sp.csr_matrix((adj_np[:, 2].astype(float), (adj_np[:, 0].astype(int), adj_np[:, 1].astype(int))))

    if algorithm == 'Leiden':
        detect_obj = Leiden(return_aggregate=False, verbose=False)
    elif algorithm == 'Louvain':
        detect_obj = Louvain(return_aggregate=False, verbose=False)

    detect_obj.fit(adj_csr, force_bipartite=force_bipartite)

    # TODO: get only labels from a user if the label probability is over 0.4, otherwise delete the user
    # TODO: if multiple labels have a probability of over 0.4, add another row of the user with the other label
    user_probs = detect_obj.probs_row_.toarray()
    item_probs = detect_obj.probs_col_.toarray()
    # argsort the probabilities to get the highest probability label
    # only take columns that have not all zeros
    user_probs = user_probs[:, np.any(user_probs, axis=0)]
    item_probs = item_probs[:, np.any(item_probs, axis=0)]

    user_probs_argsorted = np.argsort(user_probs, axis=1)[:, ::-1]
    item_probs_argsorted = np.argsort(item_probs, axis=1)[:, ::-1]

    # the detect_obj.labels doesnt always assign the most probable label for some reason so we have to do
    user_labels = user_probs_argsorted[:, 0]
    item_labels = item_probs_argsorted[:, 0]

    # keeping for community connectivity matrix
    np.savetxt(f'{save_path}/user_labels_{algorithm}_raw.csv', user_labels, delimiter=",", fmt='% 4d')
    np.savetxt(f'{save_path}/item_labels_{algorithm}_raw.csv', item_labels, delimiter=",", fmt='% 4d')

    user_probs_sorted = np.sort(user_probs, axis=1)[:, ::-1]
    item_probs_sorted = np.sort(item_probs, axis=1)[:, ::-1]

    # by visual inspection of the community interaction matrix, we choose:
    #    1 / user_probs.shape[1] * user_probs.shape[1] ** (1/3))
    #    = user_probs.shape[1] ** (-2/3)
    user_community_threshold = user_probs.shape[1] ** (-2/3)  # TODO: introduce statistical significance test instead of **(-2/3)
    item_community_threshold = item_probs.shape[1] ** (-2/3)
    # if the probability of the first label is less than (1/nr_of_labels * 1.5) , set the label to -1
    user_labels[user_probs_sorted[:, 0] < user_community_threshold] = -1
    item_labels[item_probs_sorted[:, 0] < item_community_threshold] = -1

    user_labels_matrix = np.zeros((user_labels.shape[0], user_probs_sorted.shape[1]), dtype=np.int64)
    item_labels_matrix = np.zeros((item_labels.shape[0], item_probs_sorted.shape[1]), dtype=np.int64)
    user_labels_matrix[:, 0] = user_labels
    item_labels_matrix[:, 0] = item_labels

    for i in range(1, user_probs.shape[1]):
        user_labels_i = np.where(user_probs_sorted[:, i] > user_community_threshold, user_probs_argsorted[:, i], -1)
        item_labels_i = np.where(item_probs_sorted[:, i] > item_community_threshold, item_probs_argsorted[:, i], -1)

        if np.all(user_labels_i == -1) and np.all(item_labels_i == -1):
            break
        user_labels_matrix[:, i] = user_labels_i
        item_labels_matrix[:, i] = item_labels_i

    # keep only columns that have not all zeros
    user_labels_matrix = user_labels_matrix[:, np.any(user_labels_matrix, axis=0)]
    item_labels_matrix = item_labels_matrix[:, np.any(item_labels_matrix, axis=0)]

    # used for dropout
    np.savetxt(f'{save_path}/user_labels_{algorithm}_matrix.csv', user_labels_matrix, delimiter=",", fmt='% 4d')
    np.savetxt(f'{save_path}/item_labels_{algorithm}_matrix.csv', item_labels_matrix, delimiter=",", fmt='% 4d')

    if get_probs:
        np.savetxt(f'{save_path}/user_labels_{algorithm}_probs.csv', user_probs,
                   delimiter=",", fmt='%.4f')
        np.savetxt(f'{save_path}/item_labels_{algorithm}_probs.csv', item_probs,
                   delimiter=",", fmt='%.4f')

    return torch.tensor(user_labels_matrix, dtype=torch.int64, device=device), torch.tensor(item_labels_matrix, dtype=torch.int64, device=device)


def get_power_users_items(config, adj_tens, user_com_labels, item_com_labels,
                          users_top_percent=0.01, items_top_percent=0.01,
                          save_path='/dataset/ml-32m'):
    """
    Get the indices of the top users and items based on their degree.
    :param config: dict, configuration dictionary
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
            f'dataset/{config.dataset}') or f'power_item_ids_top{items_top_percent}.csv' in os.listdir(
            f'dataset/{config.dataset}'):
        return torch.tensor(
            np.loadtxt(f'dataset/{config.dataset}/power_user_ids_top{users_top_percent}.csv'),
            dtype=torch.int64,
            device=device
        ), torch.tensor(
            np.loadtxt(f'dataset/{config.dataset}/power_item_ids_top{items_top_percent}.csv'),
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
    nr_users = max(torch.max(users).item(), user_com_labels.size(0) - 1) + 1
    nr_items = max(torch.max(items).item(), item_com_labels.size(0) - 1) + 1
    nr_user_communities = torch.max(user_com_labels).item() + 1
    nr_item_communities = torch.max(item_com_labels).item() + 1

    # Initialize accumulator sparse tensors
    item_communities_each_user = torch.zeros((nr_users, nr_item_communities), device=device)
    user_communities_each_item = torch.zeros((nr_items, nr_user_communities), device=device)

    # Get number of columns in community label matrices
    user_comm_cols = user_com_labels.size(1)
    item_comm_cols = item_com_labels.size(1)

    # Process each user community column
    for col in range(user_comm_cols):
        user_communities = user_com_labels[users, col]
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
    for col in range(item_comm_cols):
        item_communities = item_com_labels[items, col]
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

    return item_communities_each_user, user_communities_each_item





