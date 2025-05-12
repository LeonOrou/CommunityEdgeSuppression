import numpy as np
import torch
import random
import torch_geometric
# from line_profiler_pycharm import profile
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from precompute import get_user_item_community_connectivity_matrices


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
def power_node_edge_dropout(adj_tens, user_com_labels, item_com_labels, power_users_idx,
                            power_items_idx,
                            user_community_connectivity_matrix=None,
                            item_community_connectivity_matrix=None,
                            users_dec_perc_drop=0.1,
                            items_dec_perc_drop=0.2,
                            community_dropout_strength=0.6,
                            drop_from_power_nodes=True):
    # Make a copy to avoid modifying the original tensor
    adj_tens = adj_tens.clone()

    # Create a boolean mask for tracking edges to drop (more efficient than concatenating tensors)
    drop_mask = torch.zeros(adj_tens.shape[0], dtype=torch.bool, device=adj_tens.device)

    # TODO: count edges from each user and get item community for each user above the threshold

    # Process power users
    if users_dec_perc_drop > 0.0 and power_users_idx.numel() > 0 and drop_from_power_nodes:
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
                # 'user' community is in which item community the user has most connections to
                # TODO: count edges from each user and get item community for each user above the threshold
                # TODO: can be multiple! The mask goes over all communities above threshold then
                in_com_mask = ...

                # Separate edge indices by community
                in_com_indices = user_edge_indices[in_com_mask]
                out_com_indices = user_edge_indices[~in_com_mask]  # '~' means 'not'

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

    if items_dec_perc_drop > 0.0 and power_items_idx.numel() > 0 and drop_from_power_nodes:
        item_communities = item_com_labels[power_items_idx]

        for i, item in enumerate(power_items_idx):
            item_edge_indices = torch.nonzero(adj_tens[:, 1] == item).squeeze(1)

            if item_edge_indices.numel() > 0:
                user_communities = user_com_labels[adj_tens[item_edge_indices, 0]]

                item_community = item_communities[i]
                # 'item' community is in which user community the item has most connections to
                # TODO: count edges from each user and get item community for each user above the threshold
                # TODO: can be multiple! The mask goes over all communities above threshold then
                in_com_mask = ...

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

    if not drop_from_power_nodes and community_dropout_strength > 0.0:
        # drop edges from ALL users and items, not just power nodes
        total_drop_count = int(adj_tens.shape[0] * (users_dec_perc_drop + items_dec_perc_drop))
        in_com_drop_rate = users_dec_perc_drop + community_dropout_strength * (1 - users_dec_perc_drop)
        in_com_drop_count = int(adj_tens.shape[0] * in_com_drop_rate)
        out_com_drop_count = total_drop_count - in_com_drop_count
        # randomly select edges to drop
        if in_com_drop_count > 0:
            # TODO: drop in community!!
            perm = torch.randperm(adj_tens.shape[0])[:in_com_drop_count]
            drop_mask[perm] = True
        if out_com_drop_count > 0:
            # TODO: drop out of community!!
            perm = torch.randperm(adj_tens.shape[0])[:out_com_drop_count]
            drop_mask[perm] = True

    adj_tens[drop_mask, 2] = 0

    adj_tens = adj_tens[adj_tens[:, 2] != 0]

    return adj_tens


def plot_community_connectivity_distribution(connectivity_matrix, top_n_communities=10, save_path=None, dataset_name=''):
    """
    Create a line plot showing the distribution of connections from each user community to item communities,
    sorted in decreasing order.

    :param connectivity_matrix: torch.tensor, matrix showing connections between user communities (rows)
                                and item communities (columns)
    :param top_n_communities: int, number of top user communities to display
    :param save_path: str, path to save the figure, or None to display
    :param dataset_name: str, name of the dataset for saving figures
    """

    # Convert to numpy if tensor
    if isinstance(connectivity_matrix, torch.Tensor):
        connectivity_matrix = connectivity_matrix.cpu().numpy()

    # Calculate total interactions per user community for sorting
    user_community_totals = connectivity_matrix.sum(axis=1)

    # Sort user communities by total interaction volume (descending)
    sorted_indices = np.argsort(user_community_totals)[::-1]

    # Select top n user communities
    top_n = min(top_n_communities, len(sorted_indices))
    top_user_communities = sorted_indices[:top_n]

    # Create plot
    plt.figure(figsize=(12, 8))

    for i, comm_idx in enumerate(top_user_communities):
        # Normalize row to get proportion of interactions
        row_total = user_community_totals[comm_idx]
        if row_total > 0:
            normalized_row = connectivity_matrix[comm_idx, :] / row_total

            # Sort values in decreasing order
            sorted_values = np.sort(normalized_row)[::-1]

            # Create x-axis as percentage of item communities
            x_vals = np.linspace(0, 1, len(sorted_values))

            # Plot line
            plt.plot(x_vals, sorted_values,
                     label=f'Community {comm_idx} (n={int(user_community_totals[comm_idx])})')

    plt.xlabel('Proportion of Item Communities (sorted)')
    plt.ylabel('Connection Strength (normalized)')
    plt.title('User Community Connectivity Distribution (Decreasing Order)')
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
            normalized_row = connectivity_matrix[comm_idx, :] / user_community_totals[comm_idx]
            sorted_values = np.sort(normalized_row)[::-1]

            # Calculate percentage of connections in top 10% of communities
            num_top_10_percent = max(1, int(0.1 * len(sorted_values)))
            concentration_top10pct = np.sum(sorted_values[:num_top_10_percent])

            print(f"Community {comm_idx}: {concentration_top10pct * 100:.1f}% of connections in top 10% of item communities")


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
            community_confidence[comm] = np.sort(conf_scores)[::-1]  # Sort in decreasing order

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

    plt.xlabel('Proportion of Users in Community (sorted)')
    plt.ylabel('Community Assignment Confidence')
    plt.title('User Community Assignment Confidence (Decreasing Order)')
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


# via euclidean distance between user/item community connectivity matrices and uniform distribution
# for each user and each item, the bias individually
def get_community_bias(adj_tens, user_com_labels, item_com_labels):
    """
    Get the community bias of the users and items.

    :param adj_tens: torch.tensor, adjacency matrix with format (n, 3) containing (user_id, item_id, rating)
    :param user_com_labels: torch.tensor, community labels for each user
    :param item_com_labels: torch.tensor, community labels for each item
    :return: tuple of torch.tensors, community bias for users and items
    """
    device = adj_tens.device  # Get the device of the input tensor

    # Get the community connectivity matrices
    item_communities_each_user, user_communities_each_item = get_user_item_community_connectivity_matrices(
        adj_tens, user_com_labels, item_com_labels)

    # Normalize the matrices to get the probability distributions
    item_communities_each_user /= item_communities_each_user.sum(dim=1, keepdim=True)
    user_communities_each_item /= user_communities_each_item.sum(dim=1, keepdim=True)

    # Calculate the bias as the Euclidean distance from a uniform distribution
    uniform_distribution_users = torch.full_like(item_communities_each_user, 1.0 / item_communities_each_user.size(1))
    uniform_distribution_items = torch.full_like(user_communities_each_item, 1.0 / user_communities_each_item.size(1))

    # torch.norm does L2 norm by default
    user_bias = torch.linalg.matrix_norm(item_communities_each_user - uniform_distribution_users, dim=1)
    item_bias = torch.linalg.matrix_norm(user_communities_each_item - uniform_distribution_items, dim=1)

    # Normalize the bias to be between 0 and 1
    # make worst possible distribution and divide by it to make it the maximum 1
    worst_distribution_users = torch.zeros_like(item_communities_each_user)
    worst_distribution_items = torch.zeros_like(user_communities_each_item)
    worst_distribution_users[0, 0] = 1.0  # worst distribution is all in one community
    worst_distribution_items[0, 0] = 1.0

    bias_worst_users = torch.linalg.matrix_norm(item_communities_each_user - worst_distribution_users, dim=1)
    bias_worst_items = torch.linalg.matrix_norm(user_communities_each_item - worst_distribution_items, dim=1)

    user_bias /= bias_worst_users
    item_bias /= bias_worst_items

    return user_bias.cpu(), item_bias.cpu()


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


