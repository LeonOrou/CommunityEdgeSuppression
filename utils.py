import numpy as np
import torch
import pandas
import random
import scipy
import torch_geometric
from sknetwork.clustering import Leiden
import scipy.sparse as sp
import pickle
from line_profiler_pycharm import profile

# set seed function for all libraries used in the project
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_geometric.seed_everything(seed)
    # scipy.random.seed(seed)  # not needed as they use numpy seed
    # pandas.util.testing.rng = np.random.RandomState(seed)

@profile
def power_node_edge_dropout(adj_tens, user_com_labels, item_com_labels, power_users_idx, com_avg_dec_degrees,
                                     power_items_idx,
                                     users_dec_perc_drop=0.7,
                                     items_dec_perc_drop=0.3,
                                     community_dropout_strength=0.9):
    # Make a copy to avoid modifying the original tensor
    adj_tens = adj_tens.clone()

    # Create a boolean mask for tracking edges to drop (more efficient than concatenating tensors)
    drop_mask = torch.zeros(adj_tens.shape[0], dtype=torch.bool)

    # Process power users
    if users_dec_perc_drop > 0.0 and power_users_idx.numel() > 0:
        # Pre-compute user communities for all power users
        user_communities = user_com_labels[power_users_idx]

        for i, user in enumerate(power_users_idx):
            # Find edges connected to this user more efficiently
            user_edge_indices = torch.nonzero(adj_tens[:, 0] == user).squeeze(1)

            if user_edge_indices.numel() > 0:
                # Get communities of connected items
                item_communities = item_com_labels[adj_tens[user_edge_indices, 1]]

                # Identify in-community and out-of-community edges
                user_community = user_communities[i]
                in_com_mask = item_communities == user_community

                # Separate edge indices by community
                in_com_indices = user_edge_indices[in_com_mask]
                out_com_indices = user_edge_indices[~in_com_mask]

                # Calculate dropout rates and counts
                total_drop_count = int(user_edge_indices.numel() * users_dec_perc_drop)
                in_com_drop_rate = users_dec_perc_drop + community_dropout_strength * (1 - users_dec_perc_drop)
                in_com_drop_count = min(int(in_com_indices.numel() * in_com_drop_rate), in_com_indices.numel())
                out_com_drop_count = min(total_drop_count - in_com_drop_count, out_com_indices.numel())

                # Randomly select edges to drop
                if in_com_drop_count > 0:
                    perm = torch.randperm(in_com_indices.numel())[:in_com_drop_count]
                    drop_mask[in_com_indices[perm]] = True

                if out_com_drop_count > 0:
                    perm = torch.randperm(out_com_indices.numel())[:out_com_drop_count]
                    drop_mask[out_com_indices[perm]] = True

    # Process power items (similar approach)
    if items_dec_perc_drop > 0.0 and power_items_idx.numel() > 0:
        # Pre-compute item communities
        item_communities = item_com_labels[power_items_idx]

        for i, item in enumerate(power_items_idx):
            item_edge_indices = torch.nonzero(adj_tens[:, 1] == item).squeeze(1)

            if item_edge_indices.numel() > 0:
                user_communities = user_com_labels[adj_tens[item_edge_indices, 0]]

                item_community = item_communities[i]
                in_com_mask = user_communities == item_community

                in_com_indices = item_edge_indices[in_com_mask]
                out_com_indices = item_edge_indices[~in_com_mask]

                total_drop_count = int(item_edge_indices.numel() * items_dec_perc_drop)
                in_com_drop_rate = items_dec_perc_drop + community_dropout_strength * (1 - items_dec_perc_drop)
                in_com_drop_count = min(int(in_com_indices.numel() * in_com_drop_rate), in_com_indices.numel())
                out_com_drop_count = min(total_drop_count - in_com_drop_count, out_com_indices.numel())

                if in_com_drop_count > 0:
                    perm = torch.randperm(in_com_indices.numel())[:in_com_drop_count]
                    drop_mask[in_com_indices[perm]] = True

                if out_com_drop_count > 0:
                    perm = torch.randperm(out_com_indices.numel())[:out_com_drop_count]
                    drop_mask[out_com_indices[perm]] = True

    # Set ratings of dropped edges to 0
    adj_tens[drop_mask, 2] = 0

    # Filter out edges with zero ratings
    adj_tens = adj_tens[adj_tens[:, 2] != 0]

    return adj_tens


# def power_node_edge_dropout(adj_tens, user_com_labels, item_com_labels, power_users_idx, com_avg_dec_degrees,
#                             power_items=torch.tensor([]),
#                             users_dec_perc_drop=0.2,
#                             items_dec_perc_drop=0.3,
#                             community_dropout_strength=0.7):  # community_dropout_strength=0 means
#     # TODO: probably remove com_avg_dec_degrees as there is enough customization with the community dropout strength parameter, e.g. 0.8 for com strength, check this with scientific evidence
#     """
#     Drop edges of users and items that are above the threshold in their degree distribution.
#     All in torch tensor format.
#     :param com_avg_dec_degrees:
#     :param user_com_labels:
#     :param item_com_labels:
#     :param adj_tens: torch.tensor, format (n, 3) with (user, item, rating)
#     :param power_users_idx: torch.tensor, node ids of power users
#     :param power_items: torch.tensor, node ids of power items
#     :param users_dec_perc_drop: float, decimal percentage of power users' edges to drop (1 is average degree inside community)
#     :param items_dec_perc_drop: float, decimal percentage of power items' edges to drop (1 is average degree inside community)
#     :param community_dropout_strength: float, strength of dropping edges within the community (0 ... no change - normal dropout, 1 ... first only in community)
#     :return: new adj_tensor with ratings[drop_edges]=0 at dropped edges
#     """
#     # make list of edges to keep instead of dropping, its less computation (keep 1-drop)
#     # TODO: make performance analysis
#     drop_edges = torch.tensor([], dtype=torch.int32)
#     power_users_idx = power_users_idx.clone().detach()  # TODO: really necessary? Slows process down
#     if users_dec_perc_drop > 0.:  # if 1, then all edges from power nodes are dropped
#         for user in power_users_idx:  # doable without loop?
#             user_edges_idx = torch.where(torch.tensor(adj_tens[:, 0] == user), adj_tens[:, 0], 0.0).nonzero().flatten()
#             user_edges_com = user_edges_idx[user_com_labels[user] == item_com_labels[adj_tens[user_edges_idx, 1]]]
#             user_edges_out = user_edges_idx[user_com_labels[user] != item_com_labels[adj_tens[user_edges_idx, 1]]]
#             # perc_edges_in_community = np.sum(community_labels[user] == community_labels[user_edges]) / len(user_edges)
#             # 0 in_community_strength ... make normal random dropout
#             # 1 in_community_strength ... drop first only in community until in_community avg degree is reached, then out of community
#
#             # # done in main and accessible in config.variable_config_dict['power_users_avg_dec_degrees']
#             # com_label_user = community_labels[user]
#             # nr_users_in_com = torch.count_nonzero(community_labels == com_label_user)
#             # nr_edges_in_com = torch.sum(community_labels[adj_tens[:, 0]] == com_label_user)
#             # avg_degree_com_label = nr_edges_in_com / nr_users_in_com
#             # users_avg_dec_degrees = avg_degree_com_label / nr_users_in_com
#             idx_user_edges_drop_com = torch.randperm(len(user_edges_com))[:int(len(user_edges_com) * (users_dec_perc_drop + community_dropout_strength * (1-users_dec_perc_drop)))]
#             nr_to_drop = int(len(user_edges_idx) * users_dec_perc_drop)
#             nr_to_drop_in_com = len(idx_user_edges_drop_com)
#             # get indices of edges to keep outside the community
#             idx_user_edges_drop_out = torch.randperm(len(user_edges_out))[:nr_to_drop - nr_to_drop_in_com]
#
#             # user_edges_com = user_edges_com[idx_user_edges_drop_com]
#             # user_edges_out = user_edges_out[idx_user_edges_drop_out]
#             user_edges_com = user_edges_com[torch.isin(user_edges_com, user_edges_com[idx_user_edges_drop_com])]
#             user_edges_out = user_edges_out[torch.isin(user_edges_out, user_edges_out[idx_user_edges_drop_out])]
#             user_edges = torch.cat((user_edges_com, user_edges_out))
#             if drop_edges.size() == 0:
#                 drop_edges = user_edges
#             else:
#                 drop_edges = torch.cat((drop_edges, user_edges))
#
#     if items_dec_perc_drop > 0.:  # if 1, then no edges are dropped
#         for item in power_items:
#             item_edges_idx = torch.where(torch.tensor(adj_tens[:, 1] == item), adj_tens[:, 1], 0.0).nonzero().flatten()
#             item_edges_com = item_edges_idx[item_com_labels[item] == user_com_labels[adj_tens[item_edges_idx, 0]]]
#             item_edges_out = item_edges_idx[item_com_labels[item] != user_com_labels[adj_tens[item_edges_idx, 0]]]
#             # use  * (1-com_avg_dec_degrees[item_com_labels[item]]) if with reducing to avg degree
#             idx_item_drop_com = torch.randperm(len(item_edges_com))[:int(len(item_edges_com) * (((items_dec_perc_drop + community_dropout_strength * (1-items_dec_perc_drop)))))]
#             nr_to_drop = int(len(item_edges_idx) * items_dec_perc_drop)
#             nr_to_drop_in_com = len(idx_item_drop_com)
#             idx_item_drop_out = torch.randperm(len(item_edges_out))[:nr_to_drop - nr_to_drop_in_com]
#
#             item_edges_com = item_edges_com[idx_item_drop_com]
#             item_edges_out = item_edges_out[idx_item_drop_out]
#             item_edges = torch.cat([item_edges_com, item_edges_out])
#             if drop_edges.size() == 0:
#                 drop_edges = item_edges
#             else:
#                 drop_edges = torch.cat((drop_edges, item_edges))
#
#     adj_tens[:, 2][drop_edges] = 0
#     # deleting edges where rating is 0
#     adj_tens = adj_tens[adj_tens[:, 2] != 0]
#     return adj_tens


def get_community_labels(adj_np, algorithm='Leiden', save_path='data/ml-32m', get_probs=False, bipartite_connect=False):
    # check if column 0 and 1 do not intersect, no ambiguity
    # Ids start at 1 in MovieLens dataset
    max_userId = np.max(adj_np[:, 0])
    min_userId = np.min(adj_np[:, 0])
    # max_itemId = np.max(adj_np[:, 1])
    # min_itemId = np.min(adj_np[:, 1])
    if not np.all(adj_np[:, 0] != adj_np[:, 1]):
        adj_np[:, 1] = adj_np[:, 1] + max_userId + min_userId  # min_userId is 1, if its 0: no gap between new user and item ids

    # make undirected such that also items get community labels
    adj_np = np.concatenate([adj_np, adj_np[:, [1, 0, 2]]])
    adj_csr = sp.csr_matrix((adj_np[:, 2], (adj_np[:, 0], adj_np[:, 1])))

    if algorithm == 'Leiden':
        detect_obj = Leiden(modularity='Newman', return_aggregate=False, n_aggregations=-1, verbose=True)
    # elif algorithm == 'Louvain':
    #     detect_obj = Louvain(modularity='Newman', return_aggregate=False, verbose=True)
    # elif algorithm == 'KCenters':
    #     detect_obj = KCenters(n_clusters=10, center_position='row')  # maybe center_position='both' for center with user and item?
    # else:  # algorithm == 'PropagationClustering':
    #     detect_obj = PropagationClustering(n_iter=-1, weighted=False, sort_clusters=True, return_aggregate=False)

    if algorithm != 'PropagationClustering':
        if bipartite_connect:
            detect_obj.fit(adj_csr, force_bipartite=True)
        else:
            detect_obj.fit(adj_csr, force_bipartite=False)

    else:
        detect_obj.fit(adj_csr)

    if bipartite_connect:
        # ? assign items to those users labels that are majority vote the most often pointing to the item
        pass
    else: # bipartite_connect=False
        # do not connect anything as they are connected already
        # if index is a not-exising user or item, it will be max(community label) + 1, so we can ignore it as the indices are still true
        user_labels = detect_obj.labels_[0:max_userId+min_userId]  # +1 as indices start at 1
        item_labels = detect_obj.labels_[max_userId+min_userId:]

        np.savetxt(f'{save_path}/user_labels_undir_bip{bipartite_connect}_{algorithm}.csv', user_labels, delimiter=",")
        np.savetxt(f'{save_path}/item_labels_undir_bip{bipartite_connect}_{algorithm}.csv', item_labels, delimiter=",")

    # edge_labels = detect_obj.predict(adj_csr)
    if get_probs and algorithm != 'KCenters':
        # probs_col = detect_obj.probs_col_[:, :100].toarray()
        user_probs = detect_obj.probs_[:max_userId+1, :100].toarray()
        item_probs = detect_obj.probs_[max_userId+1:, :100].toarray()
        # np.savetxt(f'{save_path}/labels_col_uniq_undir_bip{bipartite_connect}_probs_{algorithm}.csv', probs_col, delimiter=",")
        np.savetxt(f'{save_path}/user_labels_undir_bip{bipartite_connect}_probs_{algorithm}.csv', user_probs, delimiter=",")
        np.savetxt(f'{save_path}/item_labels_undir_bip{bipartite_connect}_probs_{algorithm}.csv', item_probs, delimiter=",")

    #saving detect_obj
    with open(f'{save_path}/{algorithm}_obj_undir_bip{bipartite_connect}.pkl', 'wb') as f:
        pickle.dump(detect_obj, f)

    return torch.tensor(user_labels, dtype=torch.int64), torch.tensor(item_labels, dtype=torch.int64)


def percent_pointing_inside_com(adj_np, user_com_labels, item_com_labels, for_top_n_communities=10):
    """
    How many edges are pointing inside its own community?
    :param adj: scipy.sparse.csr_matrix, adjacency matrix
    :param community_labels: torch.tensor, community labels for each node
    :return: float, percentage of edges pointing inside its own community
    """
    # check if column 0 and 1 do not intersect, no ambiguity
    if not np.all(adj_np[:, 0] != adj_np[:, 1]):
        adj_np[:, 1] = adj_np[:, 1] + np.max(adj_np[:, 0]) + 1
        # make undirected such that also items get community labels
        # adj_np = np.concatenate((adj_np, adj_np[:, [1, 0, 2]]))

    if for_top_n_communities is not None:  # there are only 21 communities
        # count for the n biggest communities, how many edges are on average pointing inside its own community
        unique_labels, counts = np.unique(user_com_labels, return_counts=True)
        sort_index = np.argsort(counts)[::-1]
        unique_labels = unique_labels[sort_index]
        counts = counts[sort_index]
        top_n = unique_labels[:for_top_n_communities]
        num_user_nodes = len(user_com_labels)

        # # keep only nodes from n biggest communities
        # users = np.where(np.isin(community_labels, top_n))[0]
        # adj_np = adj_np[np.isin(adj_np[:, 0], users)]
        # adj_np = adj_np[np.isin(adj_np[:, 1], users)]
        #
        src_labels = user_com_labels[adj_np[:, 0]]
        dst_labels = item_com_labels[adj_np[:, 1]]

        for i, label in enumerate(top_n[:10]):
            # get number of edges pointing inside community
            sum_edges_com = np.sum(np.logical_and((src_labels == label), (dst_labels == label)))
            sum_in_com = np.sum(src_labels == label)
            print(f'Community {int(label)} has {np.round(sum_in_com/len(adj_np) * 100, 1)} % of edges, {np.round(counts[i]/num_user_nodes * 100, 1)} % of user nodes, and {np.round(sum_edges_com/sum_in_com * 100, 4)} % of edges pointing inside its own community')
        overall_percent = np.sum(src_labels == dst_labels) / len(adj_np)
        print(f'Overall percent of edges pointing inside community: {overall_percent}')


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


# measuring community bias of recommendations by checking how many of the recommendations are pointed inside the own community vs how many are pointed outside
from recbole.evaluator.base_metric import AbstractMetric
from recbole.utils import EvaluatorType

# TODO: make community bias metric: get recommendations and calculate how many are inside the community versus the recommendations without any modifications
# TODO: check data types of the inputs
# TODO: is that really already the bias or do I have to compare this with how well the new recommendations the user still likes, e.g. with NDCG normalization?
def get_community_bias(new_recs, standard_recs, community_labels):
    """
    Calculate the community bias of recommendations.
    :param new_recs: torch.tensor, new recommendations
    :param standard_recs: torch.tensor, standard recommendations
    :param community_labels: torch.tensor, community labels for each node
    :return: float, community bias
    """
    # get community labels of recommendations
    new_recs_com_labels = community_labels[new_recs]
    standard_recs_com_labels = community_labels[standard_recs]

    # get number of recommendations pointing inside community
    num_new_recs_in_com = torch.sum(new_recs_com_labels == community_labels[new_recs])
    num_standard_recs_in_com = torch.sum(standard_recs_com_labels == community_labels[standard_recs])
    # a negative value means a reduction in bias by that decimal percent, a positive one an increase
    return num_new_recs_in_com / num_standard_recs_in_com
