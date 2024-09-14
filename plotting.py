# plotting of community confidences
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import torch
import torch_geometric


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
    # Sort the probabilities for each node in decreasing order
    # convert sparse matrix to numpy array
    # probs_arr = probs.toarray()
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]

    # Calculate the average confidence for each rank across all nodes
    average_confidences = np.mean(sorted_probs, axis=0)
    std_confidences = np.std(sorted_probs, axis=0)

    # make bins of the max probabilities and plot in hist
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

