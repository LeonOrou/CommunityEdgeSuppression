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


def get_community_labels(adj_np, algorithm='Louvain', save_path='dataset/ml-100k', get_probs=False, bipartite_connect=True):
    ### TODO: try connecting bipartite graph with edges between user and item communities!!
    # TODO: implement bipartite and delete non-bipartite
    # check if column 0 and 1 do not intersect, no ambiguity (often the ids are two sets so they both start at 1)
    # Ids start at 1 in MovieLens dataset
    max_userId = np.max(adj_np[:, 0])
    min_userId = np.min(adj_np[:, 0])
    if not np.all(adj_np[:, 0] != adj_np[:, 1]):
        adj_np[:, 1] = adj_np[:,
                       1] + max_userId + min_userId  # min_userId is 1, if its 0: no gap between new user and item ids

    # make undirected such that also items get community labels
    adj_np = np.concatenate([adj_np, adj_np[:, [1, 0, 2]]])
    adj_csr = sp.csr_matrix((adj_np[:, 2], (adj_np[:, 0], adj_np[:, 1])))

    if algorithm == 'Leiden':
        detect_obj = Leiden(modularity='Newman', return_aggregate=False, n_aggregations=-1, verbose=True)
    elif algorithm == 'Louvain':
        detect_obj = Louvain(modularity='Newman', return_aggregate=False, verbose=True)

    detect_obj.fit(adj_csr, force_bipartite=bipartite_connect)

    # if index is a not-exising user or item, it will be max(community label) + 1, so we can ignore it as the indices are still true
    user_labels = detect_obj.labels_[0:max_userId + min_userId]
    item_labels = detect_obj.labels_[max_userId + min_userId:]

    np.savetxt(f'{save_path}/user_labels_undir_bip{bipartite_connect}_{algorithm}.csv', user_labels, delimiter=",")
    np.savetxt(f'{save_path}/item_labels_undir_bip{bipartite_connect}_{algorithm}.csv', item_labels, delimiter=",")

    if get_probs:
        # probs_col = detect_obj.probs_col_[:, :100].toarray()
        user_probs = detect_obj.probs_[:max_userId + 1, :100].toarray()
        item_probs = detect_obj.probs_[max_userId + 1:, :100].toarray()
        # np.savetxt(f'{save_path}/labels_col_uniq_undir_bip{bipartite_connect}_probs_{algorithm}.csv', probs_col, delimiter=",")
        np.savetxt(f'{save_path}/user_labels_undir_bip{bipartite_connect}_probs_{algorithm}.csv', user_probs,
                   delimiter=",")
        np.savetxt(f'{save_path}/item_labels_undir_bip{bipartite_connect}_probs_{algorithm}.csv', item_probs,
                   delimiter=",")

    with open(f'{save_path}/{algorithm}_obj_undir_bip{bipartite_connect}.pkl', 'wb') as f:
        pickle.dump(detect_obj, f)

    return torch.tensor(user_labels, dtype=torch.int64), torch.tensor(item_labels, dtype=torch.int64)


def get_community_connectivity_matrix(adj_np, user_com_labels, item_com_labels):
    """
    Get the connectivity matrix of the communities using vectorized operations.
    :param adj_np: numpy.ndarray, adjacency matrix with format (n, 3) containing (user_id, item_id, rating)
    :param user_com_labels: torch.tensor, community labels for each user
    :param item_com_labels: torch.tensor, community labels for each item
    :return: numpy.ndarray, connectivity matrix
    """
    user_indices = adj_np[:, 0].astype(int)
    item_indices = adj_np[:, 1].astype(int)

    user_communities = user_com_labels[user_indices].numpy() if isinstance(user_com_labels, torch.Tensor) else \
    user_com_labels[user_indices]
    item_communities = item_com_labels[item_indices].numpy() if isinstance(item_com_labels, torch.Tensor) else \
    item_com_labels[item_indices]

    n_user_communities = user_com_labels.max().item() + 1 if isinstance(user_com_labels,
                                                                        torch.Tensor) else user_com_labels.max() + 1
    n_item_communities = item_com_labels.max().item() + 1 if isinstance(item_com_labels,
                                                                        torch.Tensor) else item_com_labels.max() + 1

    # Use np.histogram2d to count interactions between communities
    connectivity_matrix, _, _ = np.histogram2d(
        user_communities,
        item_communities,
        bins=[n_user_communities, n_item_communities],
        range=[[0, n_user_communities - 1], [0, n_item_communities - 1]]
    )
    return connectivity_matrix  # raw int counts, not normalized


def percent_pointing_inside_com(connectivity_matrix, for_top_n_communities=10):
    """
    Analyze and visualize community connection patterns based on the connectivity matrix.

    :param connectivity_matrix: numpy.ndarray, matrix showing connections between user communities (rows)
                               and item communities (columns)
    :param for_top_n_communities: int, number of top communities to analyze
    :return: None, displays the visualization
    """
    # Check if matrix is empty
    if connectivity_matrix.size == 0:
        print("Empty connectivity matrix provided.")
        return

    # Get the number of user and item communities
    n_user_communities, n_item_communities = connectivity_matrix.shape

    # Calculate total interactions per user community for normalization
    user_community_totals = connectivity_matrix.sum(axis=1)

    # Handle zero totals to avoid division by zero
    user_community_totals[user_community_totals == 0] = 1

    # Normalize connectivity matrix (rows sum to 1)
    normalized_matrix = connectivity_matrix / user_community_totals[:, np.newaxis]

    # Sort user communities by total number of interactions (descending)
    sorted_indices = np.argsort(user_community_totals)[::-1]

    # Select top n user communities
    top_n = min(for_top_n_communities, len(sorted_indices))
    top_user_communities = sorted_indices[:top_n]

    # Extract normalized values for top communities
    top_normalized = normalized_matrix[top_user_communities]

    # For each top user community, find where its interactions are distributed
    community_interactions = []
    for i, user_comm_idx in enumerate(top_user_communities):
        # Get the distribution of interactions for this community
        distribution = normalized_matrix[user_comm_idx]

        # Record the percentage of interactions within the same community
        # Ensure the index is valid (community might interact with items only in other communities)
        same_comm_interaction = 0
        if user_comm_idx < n_item_communities:
            same_comm_interaction = distribution[user_comm_idx] * 100

        # Print detailed information
        print(f'User community {user_comm_idx}: {user_community_totals[user_comm_idx]:.0f} interactions, '
              f'{same_comm_interaction:.2f}% within same community')

        # Find top 3 item communities this user community interacts with
        top_item_indices = np.argsort(distribution)[::-1][:min(3, len(distribution))]
        for item_idx in top_item_indices:
            print(f'  - Item community {item_idx}: {distribution[item_idx] * 100:.2f}% of interactions')

        community_interactions.append(same_comm_interaction)

    # Calculate column-wise average for the selected top communities
    avg_distribution = np.mean(top_normalized, axis=0)

    # Create visualization
    plt.figure(figsize=(14, 7))

    # Create bar chart of intra-community interaction percentages
    plt.subplot(1, 2, 1)
    plt.bar(range(top_n), community_interactions, color='skyblue')
    plt.xlabel('Top User Communities (by interaction volume)')
    plt.ylabel('% of Interactions within Same Community')
    plt.title('Intra-Community Interaction Percentages')
    plt.xticks(range(top_n), [f'Comm {i}' for i in top_user_communities])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Create bar chart showing average distribution pattern
    plt.subplot(1, 2, 2)

    # Sort item communities by average interaction volume
    sorted_avg_indices = np.argsort(avg_distribution)[::-1]
    top_m = min(10, len(sorted_avg_indices))
    top_avg_indices = sorted_avg_indices[:top_m] if top_m > 0 else []

    if top_m > 0:
        plt.bar(range(len(top_avg_indices)),
                [avg_distribution[i] * 100 for i in top_avg_indices],
                color='lightcoral')
        plt.xlabel('Item Communities')
        plt.ylabel('Average % of Interactions')
        plt.title('Average Distribution of Interactions Across Top Communities')
        plt.xticks(range(len(top_avg_indices)), [f'Comm {i}' for i in top_avg_indices])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Calculate overall percentage of interactions within same community
    min_dim = min(n_user_communities, n_item_communities)
    diagonal_sum = np.sum(np.diag(connectivity_matrix[:min_dim, :min_dim]))
    total_sum = np.sum(connectivity_matrix)
    overall_percent = diagonal_sum / total_sum * 100 if total_sum > 0 else 0
    print(f'Overall percent of interactions within same community: {overall_percent:.2f}%')


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
    # Initialize empty tensors for power users and power items
    power_users_ids = torch.tensor([], dtype=torch.int64)
    power_items_ids = torch.tensor([], dtype=torch.int64)

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
                   power_users_ids.numpy(), delimiter=",")

    if items_top_percent > 0:
        np.savetxt(f'{save_path}/power_items_ids_com_wise_{do_power_nodes_from_community}_top{items_top_percent}.csv',
                   power_items_ids.numpy(), delimiter=",")

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
