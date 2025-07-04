# plotting of community confidences
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import torch
import torch_geometric


def plot_community_bias(user_biases, save_path='images/community_biases.png',
                        dataset_names=['Ml-100K', 'Ml-1M', 'LastFM']):
    """
    Plot the community biases for users and items with enhanced formatting and average display.

    :param user_biases: list of torch.tensors, community bias for users and each dataset
    :param save_path: str or None, path to save the figure
    :param dataset_names: list of str, names of the datasets
    """

    # Process and clean the data
    processed_biases = []
    averages = []

    for i, user_bias in enumerate(user_biases):
        if isinstance(user_bias, torch.Tensor):
            bias_array = user_bias.cpu().numpy()
        else:
            bias_array = user_bias

        # Remove nans from users who don't have any community (all below threshold)
        clean_bias = bias_array[~np.isnan(bias_array)]
        processed_biases.append(clean_bias)
        averages.append(np.mean(clean_bias))

    # Create figure with better size and DPI for clarity
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)

    # Create boxplot with enhanced styling
    box_plot = ax.boxplot(processed_biases,
                          labels=dataset_names,
                          patch_artist=True,
                          widths=0.7,
                          capprops=dict(linewidth=1.5),)

    # # Customize box colors
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    # set mean line orange colour
    plt.setp(box_plot['means'], color='orange', linewidth=2)

    # Customize other elements
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color='black', linewidth=1.5)

    for i, avg in enumerate(averages):
        ax.text(i + 1, -0.05, f'μ = {avg:.4f}',
                ha='center', va='top', fontsize=18, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Main plot styling
    ax.set_ylabel('Normalized Community Bias [0, 1]', fontsize=22)
    ax.set_xlabel('Datasets', fontsize=22)

    # Improve tick labels
    ax.tick_params(axis='x', which='major', labelsize=22, pad=10)
    ax.tick_params(axis='y', which='major', labelsize=22)

    # Set y-axis limits with some padding
    ax.set_ylim(-0.1, 1.05)  # Increased upper limit for annotations

    # Enhanced grid
    ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.5)

    # Add some spacing around the plot
    plt.tight_layout(pad=2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
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


def cut_cols_sparse_matrix(matrix, n_cols):
    """
    Cut the number of columns in a sparse matrix.

    Parameters:
    - matrix: Sparse matrix in CSR format.
    - n_cols: Number of columns to keep.

    Returns:
    - Sparse matrix in CSR format with the specified number of columns.
    """
    # Get the row indices, column indices, and data of the original matrix
    data, (row_indices, col_indices) = matrix.data, matrix.nonzero()

    # Keep only the data corresponding to the first n_cols columns
    mask = col_indices < n_cols
    data, row_indices, col_indices = data[mask], row_indices[mask], col_indices[mask]

    # Create the new matrix
    return sp.sparse.csr_matrix((data, (row_indices, col_indices)), shape=(matrix.shape[0], n_cols))


# Load the probability distribution of community labels
def plot_community_confidences(probs, algorithm='Leiden'):
    # Sort the rel_rec_freqs for each node in decreasing order
    # convert sparse matrix to numpy array
    # probs_arr = probs.toarray()
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]

    # Calculate the average confidence for each rank across all nodes
    average_confidences = np.mean(sorted_probs, axis=0)
    std_confidences = np.std(sorted_probs, axis=0)

    # make bins of the max rel_rec_freqs and plot in hist
    plt.figure(figsize=(10, 6))
    plt.hist(sorted_probs[:, 0], bins=50, color='blue', alpha=0.7)
    plt.title(f'Histogram of Community Label Confidences with {algorithm}')
    plt.xlabel('Confidence')
    plt.ylabel('Number of Nodes')
    plt.grid(True)
    plt.show()

    # Plot the average confidence for the top 10 ranks
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(1, 11), average_confidences[:10], yerr=std_confidences[:10], fmt='o-', color='blue')
    plt.title(f'Average Confidence of Community Labels with {algorithm}')
    plt.xlabel('Rank')
    plt.ylabel('Average Confidence')
    plt.grid(True)
    plt.show()


def plot_avg_edge_distribution(adj_index_vector):
    # plot the average degree of nodes
    # adj is sparse matrix in csr format (data, (row_indices, col_indices))
    """
        Plot the distribution of the number of nodes for each degree.

        Parameters:
        - adj: Sparse matrix in CSR format representing the adjacency matrix of the graph.
        """
    # Calculate the degree of each node
    # count the number of occurencies in the index vecotr as the degree of the node

    # # Count the number of nodes for each degree
    # unique_degrees, counts_unique = np.unique(adj_index_vector, return_counts=True)
    # # sort such that degrees in decreasing order

    # sort_index = np.argsort(counts_unique)[::-1]
    # unique_degree = unique_degrees[sort_index]
    # counts_unique = counts_unique[sort_index]
    # # add the zero degree to counts_unique
    # counts_unique = np.append(counts_unique, 0)

    p_degrees = torch_geometric.utils.degree(torch.tensor(adj_index_vector))
    # get decending order
    df = pd.DataFrame(adj_index_vector, columns=["userId"])
    p_degrees = df.groupby('userId').size()
    # get degrees from touples (always second element) in decreasing order
    # get degrees from Series object
    p_degrees = p_degrees.to_numpy()
    indices = np.argsort(p_degrees)
    p_degrees = p_degrees[indices]
    avg_degree = np.mean(p_degrees)
    print("Average degree of nodes:", avg_degree)
    n_bins = max(p_degrees) - min(p_degrees) + 1
    counts, bins = np.histogram(p_degrees, bins=n_bins)

    # plot commulative percentual of nodes with degree
    plt.figure(figsize=(10, 6))
    cum_degrees_bins = np.cumsum(counts) / np.sum(counts)
    # at which degree is the top 1, 0.5, 0.1% of nodes?
    print("Degree of top 1% of nodes:", np.where(cum_degrees_bins >= 0.99)[0][0])
    print("Degree of top 0.5% of nodes:", np.where(cum_degrees_bins >= 0.995)[0][0])
    print("Degree of top 0.1% of nodes:", np.where(cum_degrees_bins >= 0.999)[0][0])
    plt.plot(cum_degrees_bins)
    plt.title('Cumulative Distribution of Node Degrees')
    plt.xlabel('Degree')
    plt.ylabel('Cumulative Percent of Nodes')
    plt.grid(True)
    plt.show()


# ml_data = pd.read_csv("data/ml-32m/ratings_threshold.csv", delimiter=",")
# ml_data_np = ml_data.to_numpy()

# path_probs = "data/ml-32m/labels_col_probs_Leiden.csv"
# plot_community_confidences(np.loadtxt(path_probs, delimiter=","), algorithm='Leiden')
# path_probs = "data/ml-32m/labels_probs_Leiden.csv"
# plot_community_confidences(np.loadtxt(path_probs, delimiter=","), algorithm='Leiden')

