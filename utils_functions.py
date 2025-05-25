import numpy as np
import torch
import random
import torch_geometric
# from line_profiler_pycharm import profile
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import binom


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


def power_node_edge_dropout(adj_tens, power_users_idx,
                            power_items_idx,
                            biased_user_edges_mask=None,
                            biased_item_edges_mask=None,
                            users_dec_perc_drop=0.05,
                            items_dec_perc_drop=0.05,
                            community_suppression=0.6,
                            drop_only_power_nodes=True):
    """
    Drop edges from the adjacency tensor based on community labels and dropout rates.
    :param adj_tens: torch.tensor, adjacency tensor with shape (n, 3) containing (user_id, item_id, rating)
    :param power_users_idx: torch.tensor vector, adj_tens indices of power users
    :param power_items_idx: torch.tensor vector, adj_tens indices of power items
    :param biased_user_edges_mask: torch.tensor bool vector, adj_tens indices of biased user edges, i.e. edges they most interact with / in-community
    :param biased_item_edges_mask: torch.tensor bool vector, adj_tens indices of biased item edges, i.e. edges they most interact with / in-community
    :param users_dec_perc_drop: float [0, 1], dropout percentage of adj_tens for biased user edges
    :param items_dec_perc_drop: float [0, 1], dropout percentage of adj_tens for biased item edges
    :param community_suppression: float [0, 1]: float, strength of community dropout: [0, 1]; 0 means normal dropout, 1 only biased / in-community
    :param drop_only_power_nodes: bool, whether to drop edges from power nodes or not
    :return: adj_tens: torch.tensor, modified adjacency tensor without dropped edges (dropped via mask)
    """

    # Make a copy to avoid modifying the original tensor
    adj_tens = adj_tens.clone()
    device = adj_tens.device

    # Create a boolean mask for tracking edges to drop (more efficient than concatenating tensors)
    drop_mask = torch.zeros(adj_tens.shape[0], dtype=torch.bool, device=adj_tens.device)

    if users_dec_perc_drop > 0.0:
        user_edge_mask = torch.zeros(adj_tens.shape[0], dtype=torch.bool, device=device)
        if drop_only_power_nodes:
            for user_idx in power_users_idx:  # TODO: we could also precompute this and give as argument instead of power_users_idx
                user_edge_mask |= (adj_tens[:, 0] == user_idx)
        else:  # all users, not only power users
            user_edge_mask = torch.nonzero(~user_edge_mask).squeeze(1)  # ~ to get ALL edges, not just power users

        user_edge_indices = torch.nonzero(user_edge_mask).squeeze(1)
        total_user_drop_count = int(adj_tens.shape[0] * users_dec_perc_drop)  # TODO: discuss from what the users_dec_perc_drop says to drop from; all edges, biased edges, power edges

        in_com_user_indices = user_edge_indices[biased_user_edges_mask[user_edge_indices]]

        in_com_user_drop_count = min(int(total_user_drop_count), in_com_user_indices.numel())

        if in_com_user_drop_count > 0 and in_com_user_indices.numel() > 0:
            perm = torch.randperm(in_com_user_indices.numel(), device=device)[:in_com_user_drop_count]
            drop_mask[in_com_user_indices[perm]] = True

    if items_dec_perc_drop > 0.0:
        # Find all edges connected to power items
        item_edge_mask = torch.zeros(adj_tens.shape[0], dtype=torch.bool, device=device)
        if drop_only_power_nodes:
            for item_idx in power_items_idx:
                item_edge_mask |= (adj_tens[:, 1] == item_idx)
        else:  # all items, not only power items
            item_edge_mask = torch.nonzero(~item_edge_mask).squeeze(1)  # ~ to get ALL edges, not just power users

        item_edge_indices = torch.nonzero(item_edge_mask).squeeze(1)

        total_item_drop_count = int(adj_tens.shape[0] * items_dec_perc_drop)  # TODO: discuss from what the users_dec_perc_drop says to drop from; all edges, biased edges, power edges

        in_com_item_indices = item_edge_indices[biased_item_edges_mask[item_edge_indices]]

        in_com_item_drop_count = min(int(total_item_drop_count), in_com_item_indices.numel())

        if in_com_item_drop_count > 0 and in_com_item_indices.numel() > 0:
            perm = torch.randperm(in_com_item_indices.numel(), device=device)[:in_com_item_drop_count]
            drop_mask[in_com_item_indices[perm]] = True  # TODO: handle cases where the dropped edges would overlap

    adj_tens[drop_mask, 2] = community_suppression
    return adj_tens


def binomial_significance_threshold(n_iteractions, n_categories, alpha=0.05):
    """
    Returns the smallest count T such that P(X >= T) <= alpha/n_categories,
    where X ~ Binomial(n_iteractions, 1/n_categories)

    Parameters:
    - n_iteractions: total number of trials (e.g. total user interactions)
    - n_categories: number of categories (assumes uniform distribution over n_categories)
    - alpha: desired significance level (default = 0.05)

    Returns:
    - threshold: smallest integer T satisfying the inequality
    - threshold_proportion: T / n_iteractions as a float
    """
    p = 1 / n_categories
    alpha_per_test = alpha / n_categories
    for T in range(n_iteractions + 1):
        p_val = binom.sf(T - 1, n_iteractions, p)  # sf = P(X >= T)
        if p_val <= alpha_per_test:
            return T, T / n_iteractions
    return n_iteractions, 1.0  # fallback if no value found


# @profile
def plot_community_confidence(user_probs_path=None, user_labels=None, algorithm='Leiden', force_bipartite=True,
                              save_path='images/', top_n_communities=10, dataset_name=''):
    """
    Create a line plot showing community assignment confidence for each user community.

    :param user_probs_path: Path to pre-saved user probabilities CSV, or None to load based on other parameters
    :param user_labels: User community labels tensor, or None to load from save_path
    :param algorithm: Community detection algorithm used ('Leiden' or 'Louvain')
    :param force_bipartite: Whether bipartite structure was enforced
    :param save_path: Directory containing saved files
    :param top_n_communities: Number of top communities to display
    :param dataset_name: Name of the dataset for saving figures
    """

    # Load user probabilities and labels if not provided
    if user_probs_path is None:
        user_probs_path = f'{save_path}/user_labels_{algorithm}_probs.csv'

    if user_labels is None:
        user_labels = np.loadtxt(f'{save_path}/user_labels_{algorithm}.csv')
    elif isinstance(user_labels, torch.Tensor):
        user_labels = user_labels.cpu().numpy()

    # Load probability data
    user_probs = np.loadtxt(user_probs_path, delimiter=',')

    # Extract maximum probability for each user (confidence in assigned community)
    max_probs = np.max(user_probs, axis=1)

    # Group max probabilities by community
    unique_communities = np.unique(user_labels)
    community_confidence = {}

    for comm in unique_communities:
        # Get indices of users in this community
        comm_indices = np.where(user_labels == comm)[0]

        # Get confidence scores for these users
        if len(comm_indices) > 0:
            conf_scores = max_probs[comm_indices]
            community_confidence[comm] = conf_scores

    # Get communities by size (number of users)
    community_sizes = {comm: len(probs) for comm, probs in community_confidence.items()}
    top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:top_n_communities]

    # Plot confidence distributions
    plt.figure(figsize=(12, 8))

    for comm, _ in top_communities:
        # Get confidence values
        conf_values = community_confidence[comm]
        # Create normalized x-axis (percentage of users in community)
        x_vals = np.linspace(0, 1, len(conf_values))
        plt.plot(x_vals, conf_values, label=f'Community {comm} (n={len(conf_values)})')

    plt.xlabel('Item community X')
    # assign x-axis labels 1 to 4
    plt.xticks([0, 1, 2, 3], ['0', '1', '2', '3'])
    plt.ylabel('Community Assignment Confidence')
    plt.title('User Community Assignment Confidence')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Print average confidence by community
    print("\nAverage confidence by community:")
    for comm, size in top_communities:
        avg_conf = np.mean(community_confidence[comm])
        print(f"Community {comm}: {avg_conf:.4f} (n={size})")

    if save_path:
        plt.savefig(f"{save_path}/{dataset_name}_community_confidence_distribution.png")


def plot_connectivity(connectivity_matrix, save_path=None, dataset_name="", users_items='users'):
    # boxplot the connectivity columns, one boxplot for each column
    plt.figure(figsize=(12, 8))
    connectivity_matrix = connectivity_matrix[1:, :]
    # get row wise distributions
    connectivity_matrix_distribution = connectivity_matrix / connectivity_matrix.sum(axis=1, keepdims=True)
    avg_each_com = connectivity_matrix_distribution.mean(axis=0)
    std_each_com = connectivity_matrix_distribution.std(axis=0)
    # plot boxplot
    plt.boxplot(connectivity_matrix, labels=[str(i) for i in range(connectivity_matrix.shape[1])], vert=True)
    communities = 'items' if users_items == 'users' else 'users'  # reverse the labels as the users's connectivity is to item communities
    plt.xlabel(f'{communities} Community')
    plt.title('Connectivity Strength Distribution')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # also write a legend of how many connections each community has

    plt.legend([f'Community {i} (n={int(connectivity_matrix[:, i].sum())})' for i in range(connectivity_matrix.shape[1])])
    plt.show()
    if save_path:
        plt.savefig(f"{save_path}/{dataset_name}_{users_items}_connectivity_box_absolute.png")


def plot_confidence(probs, save_path=None, dataset_name="", users_items='users'):
    # boxplot the connectivity columns, one boxplot for each column
    plt.figure(figsize=(12, 8))
    # plot boxplot
    plt.boxplot(probs, labels=[str(i) for i in range(probs.shape[1])], vert=True)
    # communities = 'items' if users_items == 'users' else 'users'  # reverse the labels as the users's connectivity is to item communities
    plt.xlabel(f'{users_items} Community')
    plt.title('Community confidence')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # also write a legend of how many connections each community has

    plt.show()
    if save_path:
        plt.savefig(f"{save_path}/{dataset_name}_{users_items}_probs_confidence.png")


def plot_community_connectivity_distribution(user_connectivity_matrix, top_n_communities=10, save_path=None, dataset_name=''):
    """
    Create a line plot showing the distribution of connections from each user community to item communities,
    sorted in decreasing order.

    :param user_connectivity_matrix: torch.tensor, matrix showing connections between user communities (rows)
                                and item communities (columns)
    :param top_n_communities: int, number of top user communities to display
    :param save_path: str, path to save the figure, or None to display
    :param dataset_name: str, name of the dataset for saving figures
    """

    # Convert to numpy if tensor
    if isinstance(user_connectivity_matrix, torch.Tensor):
        user_connectivity_matrix = user_connectivity_matrix.cpu().numpy()

    # Calculate total interactions per user community for sorting
    user_community_totals = user_connectivity_matrix.sum(axis=1).T

    # Sort user communities by total interaction volume (descending)
    sorted_indices = np.argsort(user_community_totals)[::-1]

    # Select top n user communities
    top_n = min(top_n_communities, len(sorted_indices))
    top_user_communities = np.arange(user_connectivity_matrix.shape(1))

    # Create plot
    plt.figure(figsize=(12, 8))

    for i, comm_idx in enumerate(top_user_communities):
        # Skip empty communities
        if user_community_totals[comm_idx] == 0:
            continue
        # Normalize row to get proportion of interactions
        row_total = user_community_totals[comm_idx]
        if row_total > 0:
            normalized_row = user_connectivity_matrix[comm_idx, :] / row_total

            # # Sort values in decreasing order
            # sorted_values = np.sort(normalized_row)[::-1]

            # x vals are the indices of the item communities (integers)
            x_vals = np.linspace(0, 3, len(normalized_row))

            # Plot line
            plt.plot(x_vals, normalized_row,
                     label=f'Community {comm_idx} (n={int(user_community_totals[comm_idx])})')

    plt.xlabel('')
    # assign x-axis labels 1 to 4
    plt.xticks(np.arange(len(top_user_communities)), [str(i) for i in range(len(top_user_communities))])
    plt.ylabel('Connection Strength (normalized)')
    plt.title('User Community Connectivity Distribution for each item community')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/{dataset_name}_community_connectivity_distribution.png")
    plt.show()

    # Calculate concentration metrics
    print("Community concentration metrics:")
    for i, comm_idx in enumerate(top_user_communities):
        if user_community_totals[comm_idx] > 0:
            normalized_row = user_connectivity_matrix[comm_idx, :] / user_community_totals[comm_idx]
            sorted_values = np.sort(normalized_row)[::-1]

            # Calculate percentage of connections in top 10% of communities
            num_top_10_percent = max(1, int(0.1 * len(sorted_values)))
            concentration_top10pct = np.sum(sorted_values[:num_top_10_percent])

            print(f"Community {comm_idx}: {concentration_top10pct * 100:.1f}% of connections in top 10% of item communities")


# via euclidean distance between user/item community connectivity matrices and uniform distribution
# for each user and each item, the bias individually
def get_community_bias(item_communities_each_user_dist=None, user_communities_each_item_dist=None):
    """
    Get the community bias of the users and items.

    :param item_communities_each_user_dist: torch.tensor, item community distribution for each user
    :param user_communities_each_item_dist: torch.tensor, user community distribution for each item
    :return: tuple of torch.tensors, community bias for users and items
    """

    uniform_distribution_users = torch.full_like(item_communities_each_user_dist, 1.0 / item_communities_each_user_dist.size(1))
    uniform_distribution_items = torch.full_like(user_communities_each_item_dist, 1.0 / user_communities_each_item_dist.size(1))

    # torch.norm does L2 norm by default
    user_bias = torch.linalg.norm(uniform_distribution_users - item_communities_each_user_dist, dim=1)
    item_bias = torch.linalg.norm(uniform_distribution_items - user_communities_each_item_dist, dim=1)

    # Normalize the bias to be between 0 and 1
    # make worst possible distribution and divide by it to make it the maximum 1
    worst_distribution_users = torch.zeros_like(item_communities_each_user_dist)
    worst_distribution_items = torch.zeros_like(user_communities_each_item_dist)
    worst_distribution_users[:, 0] = 1.0  # worst distribution is all in one community
    worst_distribution_items[:, 0] = 1.0

    # bias for each user and item, can be processed for distributions, averages, etc.
    bias_worst_users = torch.linalg.norm(uniform_distribution_users - worst_distribution_users, dim=1)
    bias_worst_items = torch.linalg.norm(uniform_distribution_items - worst_distribution_items, dim=1)

    user_bias /= bias_worst_users
    item_bias /= bias_worst_items

    return user_bias.cpu(), item_bias.cpu()


def plot_community_bias(user_biases, item_biases, save_path=None, dataset_name=''):
    """
    Plot the community biases for users and items.

    :param user_biases: torch.tensor, community bias for users
    :param item_biases: torch.tensor, community bias for items
    :param save_path: str or None, path to save the figure
    :param dataset_name: str, name of the dataset for saving figures
    """
    # index zero is not a node, so we need to remove it
    user_biases = user_biases[1:]
    item_biases = item_biases[1:]

    # Convert to numpy if tensor
    if isinstance(user_biases, torch.Tensor):
        user_biases = user_biases.cpu().numpy()
        item_biases = item_biases.cpu().numpy()

    plt.figure(figsize=(12, 8))
    plt.boxplot([user_biases, item_biases], labels=['User bias', 'Items bias'])
    plt.title('Community Bias for each user and item')
    plt.ylabel('Bias [0, 1]')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(f"{save_path}/{dataset_name}_community_bias.png")
    plt.show()


def plot_degree_distributions(adj_tens, num_bins=100, save_path=None, dataset_name=''):
    """
    Plot the degree distributions for users and items in decreasing order.

    :param adj_tens: torch.tensor, adjacency matrix with format (n, 3) containing (user_id, item_id, rating)
    :param num_bins: int, number of percentile bins to use
    :param save_path: str or None, path to save the figures
    :param dataset_name: str, name of the dataset for saving figures
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # Convert to numpy if it's a tensor
    if isinstance(adj_tens, torch.Tensor):
        adj_np = adj_tens.cpu().numpy()
    else:
        adj_np = adj_tens

    # Extract user and item indices
    user_indices = adj_np[:, 0].astype(int)
    item_indices = adj_np[:, 1].astype(int)

    # Get unique nodes and their degrees
    unique_users, user_counts = np.unique(user_indices, return_counts=True)
    unique_items, item_counts = np.unique(item_indices, return_counts=True)

    # Sort degrees in descending order
    sorted_user_degrees = np.sort(user_counts)[::-1]
    sorted_item_degrees = np.sort(item_counts)[::-1]

    # Create percentile bins for node rankings
    user_percentiles = np.linspace(0, 100, num_bins + 1)
    item_percentiles = np.linspace(0, 100, num_bins + 1)

    # Use percentiles to get indices into sorted arrays
    user_bin_indices = np.percentile(np.arange(len(sorted_user_degrees)), user_percentiles).astype(int)
    item_bin_indices = np.percentile(np.arange(len(sorted_item_degrees)), item_percentiles).astype(int)

    # Make sure the last index points to the end of the array
    user_bin_indices[-1] = len(sorted_user_degrees)
    item_bin_indices[-1] = len(sorted_item_degrees)

    # Calculate average degrees for each percentile bin
    user_bin_degrees = [np.mean(sorted_user_degrees[user_bin_indices[i]:user_bin_indices[i + 1]])
                        for i in range(len(user_bin_indices) - 1)]
    item_bin_degrees = [np.mean(sorted_item_degrees[item_bin_indices[i]:item_bin_indices[i + 1]])
                        for i in range(len(item_bin_indices) - 1)]

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot user degree distribution
    ax1.bar(range(len(user_bin_degrees)), user_bin_degrees)
    ax1.set_title('User Degree Distribution (Decreasing Order)')
    ax1.set_xlabel('Percentile Bin')
    ax1.set_ylabel('Average Degree')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Add user statistics
    ax1.text(0.02, 0.95,
             f"Users: {len(unique_users)}\n"
             f"Max degree: {sorted_user_degrees[0]}\n"
             f"Mean degree: {np.mean(user_counts):.1f}\n"
             f"Median degree: {np.median(user_counts):.1f}",
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # Plot item degree distribution
    ax2.bar(range(len(item_bin_degrees)), item_bin_degrees)
    ax2.set_title('Item Degree Distribution (Decreasing Order)')
    ax2.set_xlabel('Percentile Bin')
    ax2.set_ylabel('Average Degree')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add item statistics
    ax2.text(0.02, 0.95,
             f"Items: {len(unique_items)}\n"
             f"Max degree: {sorted_item_degrees[0]}\n"
             f"Mean degree: {np.mean(item_counts):.1f}\n"
             f"Median degree: {np.median(item_counts):.1f}",
             transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/{dataset_name}_degree_distribution_info.png")
    plt.show()

    # Create line plot of actual degree distributions
    plt.figure(figsize=(16, 8))

    # Create normalized x-axis
    user_x = np.linspace(0, 1, len(sorted_user_degrees))
    item_x = np.linspace(0, 1, len(sorted_item_degrees))

    plt.plot(user_x, sorted_user_degrees, label=f'Users ({len(unique_users)})')
    plt.plot(item_x, sorted_item_degrees, label=f'Items ({len(unique_items)})')

    plt.xlabel('Normalized Node Rank')
    plt.ylabel('Degree')
    plt.title('Node Degree Distribution (Decreasing Order)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(f"{save_path}/{dataset_name}_degree_distribution.png")
    plt.show()

    # Create log-scale version for better visualization of tail distribution
    plt.figure(figsize=(16, 8))
    plt.semilogy(user_x, sorted_user_degrees, label=f'Users ({len(unique_users)})')
    plt.semilogy(item_x, sorted_item_degrees, label=f'Items ({len(unique_items)})')

    plt.xlabel('Normalized Node Rank')
    plt.ylabel('Degree (Log Scale)')
    plt.title('Node Degree Distribution (Decreasing Order, Log Scale)')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(f"{save_path}_log.png")
    plt.show()


