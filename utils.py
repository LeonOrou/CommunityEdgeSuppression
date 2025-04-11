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

# @profile
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
