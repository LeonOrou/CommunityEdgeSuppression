import numpy as np
import torch
import pandas
import random
import scipy
import torch_geometric
from sknetwork.clustering import Leiden
import scipy.sparse as sp

# set seed function for all libraries used in the project
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch_geometric.seed_everything(seed)
    # scipy.random.seed(seed)  # not needed as they use numpy seed
    # pandas.util.testing.rng = np.random.RandomState(seed)


def power_node_edge_dropout(adj_tens, community_labels, power_users_idx, power_items=[], users_dec_perc_drop=0.7, items_dec_perc_drop=0.0, community_dropout_strength=0):
    """
    Drop edges of users and items that are above the threshold in their degree distribution.
    All in torch tensor format.
    :param adj_tens: torch.tensor, format (n, 3) with (user, item, rating)
    :param community_labels: torch.tensor, community labels for each node id community_labels[user_id] = community_label
    :param power_users: torch.tensor, node ids of power users
    :param power_items: torch.tensor, node ids of power items
    :param user_perc_drop: float, decimal percentage of power users' edges to drop (1 is average degree inside community)
    :param item_perc_drop: float, decimal percentage of power items' edges to drop (1 is average degree inside community)
    :param community_dropout_strength: float, strength of dropping edges within the community (0 ... no change, 1 ... first only in community)
    :return: new adj_tensor with ratings[drop_edges]=0 at dropped edges
    """
    # adj_new = adj.copy()
    # empty list to store dropped edges
    drop_edges = torch.tensor([], dtype=torch.int32)
    power_users_idx = power_users_idx.clone().detach()
    if users_dec_perc_drop > 0.:  # if 1, then no edges are dropped
        for user in power_users_idx:
            user_edges_idx = torch.where(adj_tens[:, 0] == user, adj_tens[:, 0], 0.0).nonzero().flatten()
            user_edges_com = user_edges_idx[community_labels[user] == community_labels[adj_tens[user_edges_idx, 1]]]
            user_edges_out = user_edges_idx[community_labels[user] != community_labels[adj_tens[user_edges_idx, 1]]]
            # perc_edges_in_community = np.sum(community_labels[user] == community_labels[user_edges]) / len(user_edges)
            # 0 in_community_strength ... make normal random dropout
            # 1 in_community_strength ... drop first only in community until in_community avg degree is reached, then out of community
            nr_to_drop = int(len(user_edges_idx) * users_dec_perc_drop)
            com_label_user = community_labels[user]
            nr_users_in_com = torch.count_nonzero(community_labels == com_label_user)
            nr_edges_in_com = torch.sum(community_labels[adj_tens[:, 0]] == com_label_user)
            avg_degree_com_label = nr_edges_in_com / nr_users_in_com
            decimal_percent_until_avg_degree = avg_degree_com_label / nr_users_in_com

            idx_user_edges_com = torch.randint(0, len(user_edges_com), (int(len(user_edges_com) * ((users_dec_perc_drop + community_dropout_strength * (1-users_dec_perc_drop)) * (1-decimal_percent_until_avg_degree))),))
            nr_to_drop_in_com = len(idx_user_edges_com)
            idx_user_edges_out = torch.randint(0, len(user_edges_out), (nr_to_drop - nr_to_drop_in_com,))

            user_edges_com = user_edges_com[idx_user_edges_com]
            user_edges_out = user_edges_out[idx_user_edges_out]
            user_edges = torch.cat((user_edges_com, user_edges_out))
            if drop_edges.size() == 0:
                drop_edges = user_edges
            else:
                drop_edges = torch.cat((drop_edges, user_edges))

    if items_dec_perc_drop > 0.:  # if 1, then no edges are dropped
        for item in power_items:
            item_edges_idx = torch.where(adj_tens[:, 1] == item, adj_tens[:, 1], 0.0).nonzero().flatten()
            item_edges_com = item_edges_idx[community_labels[item] == community_labels[adj_tens[item_edges_idx, 0]]]
            item_edges_out = item_edges_idx[community_labels[item] != community_labels[adj_tens[item_edges_idx, 0]]]
            # perc_edges_in_community = np.sum(community_labels[item] == community_labels[item_edges]) / len(item_edges)
            # 0 in_community_strength ... make normal random dropout
            # 1 in_community_strength ... drop only in community
            nr_to_drop = int(len(item_edges_idx) * items_dec_perc_drop)
            avg_degree_in_com = np.sum(community_labels[item] == community_labels[adj_tens[item_edges_idx, 0]]) / len(item_edges_idx)
            decimal_percent_until_avg_degree = avg_degree_in_com / len(item_edges_idx)
            idx_item_edges_com = torch.randint(0, len(item_edges_com), (int(len(item_edges_com) * ((items_dec_perc_drop + community_dropout_strength * (1-items_dec_perc_drop)) * (1-decimal_percent_until_avg_degree))),))
            nr_to_drop_in_com = len(idx_item_edges_com)
            idx_item_edges_out = torch.randint(0, len(item_edges_out), (nr_to_drop - nr_to_drop_in_com,))

            item_edges_com = item_edges_com[idx_item_edges_com]
            item_edges_out = item_edges_out[idx_item_edges_out]
            item_edges_idx = torch.concatenate([item_edges_com, item_edges_out])
            if drop_edges.size() == 0:
                drop_edges = item_edges_idx
            else:
                drop_edges = torch.cat((drop_edges, item_edges_idx))

    drop_edges = torch.unique(torch.flatten(drop_edges))
    # new_edges = torch.where(torch.logical_not(torch.isin(torch.arange(adj_tens.shape[0]), drop_edges)))[0]

    # set all dropped edges to 0: new adj needs to have same dimensions for Dataset.dataset.copy(inter_feat)
    adj_tens[:, 2][drop_edges] = 0
    # deleting edges where rating is 0
    adj_tens = adj_tens[adj_tens[:, 2] != 0]
    # adj_sparse = sp.csr_matrix((np.ones(len(new_edges)), (new_edges[:, 0], new_edges[:, 1])))
    return adj_tens


def get_community_labels(adj_np, algorithm='Leiden', save_path='data/ml-32m', get_probs=False):
    # check if column 0 and 1 do not intersect, no ambiguity
    if not np.all(adj_np[:, 0] != adj_np[:, 1]):
        adj_np[:, 1] = adj_np[:, 1] + np.max(adj_np[:, 0]) + 1
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
        detect_obj.fit(adj_csr, force_bipartite=False)
    else:
        detect_obj.fit(adj_csr)

    # np.savetxt(f'{save_path}/labels_col_uniq_undir_nonbip_{algorithm}.csv', detect_obj.labels_col_, delimiter=",")
    np.savetxt(f'{save_path}/labels_uniq_undir_nonbip_{algorithm}.csv', detect_obj.labels_, delimiter=",")

    # edge_labels = detect_obj.predict(adj_csr)
    if get_probs and algorithm != 'KCenters':
        # probs_col = detect_obj.probs_col_[:, :100].toarray()
        probs = detect_obj.probs_[:, :100].toarray()
        # np.savetxt(f'{save_path}/labels_col_uniq_undir_nonbip_probs_{algorithm}.csv', probs_col, delimiter=",")
        np.savetxt(f'{save_path}/labels_uniq_undir_nonbip_probs_{algorithm}.csv', probs, delimiter=",")

    return detect_obj.labels_


def percent_pointing_inside_com(adj_np, community_labels, for_top_n_communities=10):
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
        unique_labels, counts = np.unique(community_labels, return_counts=True)
        sort_index = np.argsort(counts)[::-1]
        unique_labels = unique_labels[sort_index]
        counts = counts[sort_index]
        top_n = unique_labels[:for_top_n_communities]
        num_nodes = np.unique((adj_np[:, :2].flatten())).shape[0]
        # # get number of communities with only 1 node
        # one_counts = np.sum(counts == 1)
        # print(f'Overall % of nodes with only 1 community: {np.round(one_counts / num_nodes * 100, 2)} %')

        # # keep only nodes from n biggest communities
        # users = np.where(np.isin(community_labels, top_n))[0]
        # adj_np = adj_np[np.isin(adj_np[:, 0], users)]
        # adj_np = adj_np[np.isin(adj_np[:, 1], users)]
        #
        src_labels = community_labels[adj_np[:, 0]]
        dst_labels = community_labels[adj_np[:, 1]]

        for i, label in enumerate(top_n[:10]):
            # get number of edges pointing inside community
            # num of edges inside community
            # take all edges that have community 'label' as destination
            sum_edges_com_six = np.sum(np.logical_and((src_labels == label), (dst_labels == label)))
            sum_in_com = np.sum(src_labels == label)
            print(f'Community {int(label)} has {np.round(sum_in_com/len(adj_np) * 100, 1)} % of edges, {np.round(counts[i]/num_nodes * 100, 1)} % of nodes, and {np.round(sum_edges_com_six/sum_in_com * 100, 4)} % of edges pointing inside its own community')
        overall_percent = np.sum(src_labels == dst_labels) / len(adj_np)
        print(f'Overall percent of edges pointing inside community: {overall_percent}')


def get_power_users_items(adj_tens, community_labels, users_top_percent=0.01, items_top_percent=0, do_power_nodes_from_community=False, save_path='/data/ml-32m'):
    """
    Get the indices of the top users and items based on their degree.
    :param adj_tens: torch.tensor, format (n, 3) with (user, item, rating)
    :param users_top_percent: float, percentage of top users to keep
    :param items_top_percent: float, percentage of top items to keep
    :param community_labels: torch.tensor, community labels for each node
    :param do_power_nodes_from_community: bool, if True, get top users and items for each community
    :return: torch.tensor, indices of top users, torch.tensor, indices of top items
    """
    if do_power_nodes_from_community:
        power_users_ids = np.array([])
        power_items_ids = np.array([])
        # get top communities
        unique_labels, count = np.unique(community_labels, return_counts=True)
        sort_index = np.argsort(count)[::-1]
        # unique_labels = unique_labels[sort_index]
        count = count[sort_index]
        top_x_percent_labels_idx = np.where(count >= 0.1 * len(community_labels))[0]  # loop through all community labels that have at least 10% of the users

        for label in top_x_percent_labels_idx:
            # get all users in community
            users_idx = np.where(community_labels == label)[0]
            # get highest degrees of users
            # get edges per user
            user_edges = adj_tens[np.isin(adj_tens[:, 0], users_idx)]
            p_degrees = torch_geometric.utils.degree(torch.tensor(user_edges[:, 0]))
            p_degrees = p_degrees.numpy()
            top_users_idx = np.argsort(p_degrees)[::-1]
            # get indices of top x percent of users with the highest degree
            top_x_percent_idx = int(len(users_idx) * users_top_percent) + 1  # to have always at least 1 user from every big community
            if power_users_ids.size == 0:
                power_users_ids = top_users_idx[:top_x_percent_idx].flatten()
            else:
                power_users_ids = np.concatenate((power_users_ids, top_users_idx[:top_x_percent_idx].flatten()))
    else:
        p_degrees = torch_geometric.utils.degree(torch.tensor(adj_tens[:, 0]))
        p_degrees = p_degrees.numpy()
        top_users_idx = np.argsort(p_degrees)[::-1]
        # get indices of top x percent of users with the highest degree
        top_x_percent_idx = int(len(adj_tens) * (1 - users_top_percent)) + 1  # to have always at least 1 user from every big community
        power_users_ids = top_users_idx[:top_x_percent_idx].flatten()
    # power_users_adj_idx = np.where(np.isin(adj_tens[:, 0], power_users))
    # save indices of top users locally
    power_users_ids = np.unique(np.array(power_users_ids, dtype=np.int64))
    # also make for items when needed
    np.savetxt(f'{save_path}/power_nodes_ids.csv', power_users_ids, delimiter=",")
    return power_users_ids




