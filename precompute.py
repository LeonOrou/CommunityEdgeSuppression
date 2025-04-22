import pickle
import numpy as np
import torch
import torch_geometric
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sknetwork.clustering import Leiden, Louvain
from argparse import ArgumentParser


# get adj_np like in main.py
parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='ml-100k')
parser.add_argument("--save_path", type=str, default='data/ml-100k')

args = parser.parse_args()

dataset_name = args.dataset_name
save_path = args.save_path

# load only train data as we know only for train data


def get_community_labels(adj_np, algorithm='Leiden', save_path='dataset/ml-100k', get_probs=True, force_bipartite=True):
    ### TODO: make it work with Louvain algorithm
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
        detect_obj = Leiden(return_aggregate=False, verbose=True)
    elif algorithm == 'Louvain':
        detect_obj = Louvain(return_aggregate=False, verbose=True)

    detect_obj.fit(adj_csr, force_bipartite=force_bipartite)

    # if index is a not-exising user or item, it will be max(community label) + 1, so we can ignore it as the indices are still true
    user_labels = detect_obj.labels_row_
    item_labels = detect_obj.labels_col_  # adding max_userId to item_labels not necessary as connectivity matrix sees them as two different axes

    np.savetxt(f'{save_path}/user_labels_undir_bip{force_bipartite}_{algorithm}.csv', user_labels, delimiter=",")
    np.savetxt(f'{save_path}/item_labels_undir_bip{force_bipartite}_{algorithm}.csv', item_labels, delimiter=",")

    if get_probs:
        user_probs = detect_obj.probs_row_[:, :10].toarray()
        # adding max_userId to item_probs is not necessary as connectivity matrix sees them as two different axes
        item_probs = detect_obj.probs_col_[:, :10].toarray()
        np.savetxt(f'{save_path}/user_labels_dir_bip{force_bipartite}_probs_{algorithm}.csv', user_probs,
                   delimiter=",")
        np.savetxt(f'{save_path}/item_labels_dir_bip{force_bipartite}_probs_{algorithm}.csv', item_probs,
                   delimiter=",")

    with open(f'{save_path}/{algorithm}_obj_dir_bip{force_bipartite}.pkl', 'wb') as f:
        pickle.dump(detect_obj, f)

    return torch.tensor(user_labels, dtype=torch.int64), torch.tensor(item_labels, dtype=torch.int64)


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


def get_power_users_items(adj_tens, user_com_labels, item_com_labels,
                          users_top_percent=0.01, items_top_percent=0.01,
                          do_power_nodes_from_community=True, save_path='/data/ml-32m'):
    """
    Get the indices of the top users and items based on their degree.
    :param adj_tens: torch.tensor, format (n, 3) with (user, item, rating)
    :param user_com_labels: torch.tensor, community labels for each user
    :param item_com_labels: torch.tensor, community labels for each item
    :param users_top_percent: float, percentage of top users to keep
    :param items_top_percent: float, percentage of top items to keep
    :param do_power_nodes_from_community: bool, if True, get top users and items for each community
    :param save_path: str, path to save the output files
    :return: torch.tensor, indices of top users, torch.tensor, indices of top items
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # changed code: define device
    power_users_ids = torch.tensor([], dtype=torch.int64, device=device)  # changed code
    power_items_ids = torch.tensor([], dtype=torch.int64, device=device)  # changed code

    if do_power_nodes_from_community:
        # Process users by community
        if users_top_percent > 0:
            # Get significant communities (containing at least 1% of users)
            unique_user_labels, user_count = torch.unique(user_com_labels, return_counts=True)
            all_users = torch.unique(adj_tens[:, 0])
            top_user_com_idx = torch.where(user_count >= 0.01 * len(all_users))[0]

            for user_label in unique_user_labels[top_user_com_idx]:
                # Get users from this community
                users_idx = torch.where(user_com_labels == user_label)[0]
                # From these node ids, get the edges
                user_edges = adj_tens[torch.isin(adj_tens[:, 0], users_idx)]
                # Get highest degrees of users
                user_degrees = torch_geometric.utils.degree(user_edges[:, 0])
                sorted_users_idx = torch.argsort(user_degrees, descending=True)
                top_x_percent_user_idx = int(len(users_idx) * users_top_percent) + 1
                top_users = sorted_users_idx[:top_x_percent_user_idx].flatten()
                power_users_ids = torch.cat((power_users_ids, top_users)) if power_users_ids.size(0) > 0 else top_users

        # Process items by community
        if items_top_percent > 0:
            # Get significant communities (containing at least 1% of items)
            unique_item_labels, item_count = torch.unique(item_com_labels, return_counts=True)
            all_items = torch.unique(adj_tens[:, 1])
            top_item_com_idx = torch.where(item_count >= 0.01 * len(all_items))[0]

            for item_label in unique_item_labels[top_item_com_idx]:
                # Get items from this community
                items_idx = torch.where(item_com_labels == item_label)[0]
                # From these node ids, get the edges
                item_edges = adj_tens[torch.isin(adj_tens[:, 1], items_idx)]
                # Get highest degrees of items
                item_degrees = torch_geometric.utils.degree(item_edges[:, 1])
                sorted_items_idx = torch.argsort(item_degrees, descending=True)
                top_x_percent_item_idx = int(len(items_idx) * items_top_percent) + 1
                top_items = sorted_items_idx[:top_x_percent_item_idx].flatten()
                power_items_ids = torch.cat((power_items_ids, top_items)) if power_items_ids.size(0) > 0 else top_items
    else:
        # Global approach - get top users and items regardless of community
        if users_top_percent > 0:
            user_degrees = torch_geometric.utils.degree(adj_tens[:, 0])
            sorted_users_idx = torch.argsort(user_degrees, descending=True)
            top_x_percent_user_idx = int(len(torch.unique(adj_tens[:, 0])) * users_top_percent) + 1
            power_users_ids = sorted_users_idx[:top_x_percent_user_idx].flatten()

        if items_top_percent > 0:
            item_degrees = torch_geometric.utils.degree(adj_tens[:, 1])
            sorted_items_idx = torch.argsort(item_degrees, descending=True)
            top_x_percent_item_idx = int(len(torch.unique(adj_tens[:, 1])) * items_top_percent) + 1
            power_items_ids = sorted_items_idx[:top_x_percent_item_idx].flatten()

    # Ensure uniqueness
    power_users_ids = torch.unique(power_users_ids)
    power_items_ids = torch.unique(power_items_ids)

    # Save to files
    if users_top_percent > 0:
        np.savetxt(f'{save_path}/power_users_ids_com_wise_{do_power_nodes_from_community}_top{users_top_percent}.csv',
                   power_users_ids.cpu().numpy(), delimiter=",")

    if items_top_percent > 0:
        np.savetxt(f'{save_path}/power_items_ids_com_wise_{do_power_nodes_from_community}_top{items_top_percent}.csv',
                   power_items_ids.cpu().numpy(), delimiter=",")

    return power_users_ids, power_items_ids

# def get_power_users_items(adj_tens, user_com_labels, item_com_labels=[], users_top_percent=0.01, items_top_percent=0,
#                           do_power_nodes_from_community=False, save_path='/data/ml-32m'):
#     """
#     Get the indices of the top users and items based on their degree.
#     :param adj_tens: torch.tensor, format (n, 3) with (user, item, rating)
#     :param users_top_percent: float, percentage of top users to keep
#     :param items_top_percent: float, percentage of top items to keep
#     :param community_labels: torch.tensor, community labels for each node
#     :param do_power_nodes_from_community: bool, if True, get top users and items for each community
#     :return: torch.tensor, indices of top users, torch.tensor, indices of top items
#     """
#     if do_power_nodes_from_community:
#         power_users_ids = torch.tensor([], dtype=torch.int64)
#         # power_items_ids = torch.tensor([], dtype=torch.int64)
#         # get top communities
#         unique_labels, count = torch.unique(user_com_labels, return_counts=True)
#         user_labels = user_com_labels[torch.unique(adj_tens[:, 0])]
#         # item_labels = item_com_labels[torch.unique(adj_tens[:, 1])]
#         # get the unique_label where the label count is at least 1% of the users
#         top_1_perc_user_labels_idx = torch.where(count >= 0.01 * len(user_labels))[0]
#
#         for user_label in top_1_perc_user_labels_idx:
#             # get users from this community label
#             users_idx = torch.where(torch.tensor(user_labels == user_label))[0]
#             # from these node ids, get the edges for each id
#             user_edges = adj_tens[torch.where(torch.isin(adj_tens[:, 0], users_idx))]
#             # get highest degrees of users
#             # get edges per user
#             p_degrees = torch_geometric.utils.degree(user_edges[:, 0])
#             top_users_idx = torch.argsort(p_degrees, descending=True)
#             top_x_percent_idx = int(len(users_idx) * users_top_percent) + 1
#             if power_users_ids.size(0) == 0:
#                 power_users_ids = top_users_idx[:top_x_percent_idx].flatten()
#             else:
#                 power_users_ids = torch.cat((power_users_ids, top_users_idx[:top_x_percent_idx].flatten()))
#     else:
#         p_degrees = torch_geometric.utils.degree(adj_tens[:, 0])
#         top_users_idx = torch.argsort(p_degrees, descending=True)
#         top_x_percent_idx = int(len(adj_tens) * (1 - users_top_percent)) + 1
#         power_users_ids = top_users_idx[:top_x_percent_idx].flatten()
#
#     power_users_ids = torch.unique(power_users_ids)
#     np.savetxt(f'{save_path}/power_nodes_ids_com_wise_{do_power_nodes_from_community}_top{users_top_percent}users.csv', power_users_ids.numpy(), delimiter=",")
#     return power_users_ids
