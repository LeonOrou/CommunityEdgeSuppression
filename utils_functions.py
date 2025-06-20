import numpy as np
import torch
import random
import torch_geometric
# from line_profiler_pycharm import profile
import matplotlib
import matplotlib.pyplot as plt
import wandb
import os
import sknetwork
from precompute import get_community_labels, get_power_users_items, get_biased_edges_mask, get_user_item_community_connectivity_matrices
matplotlib.use('agg')


# set seed function for all libraries used in the project
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    sknetwork.utils.check_random_state(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_geometric.seed_everything(seed)
    # scipy.random.seed(seed)  # not needed as they use numpy seed
    # pandas.util.testing.rng = np.random.RandomState(seed)


def community_edge_suppression(adj_tens, config):
    """
    Drop edges from the adjacency tensor based on community labels and dropout rates.
    :param adj_tens: torch.tensor, adjacency tensor with shape (n, 3) containing (user_id, item_id, rating)
    :param config: Config object containing all necessary parameters
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

    power_users_idx = config.power_users_ids
    power_items_idx = config.power_items_ids
    biased_user_edges_mask = config.biased_user_edges_mask
    biased_item_edges_mask = config.biased_item_edges_mask
    drop_only_power_nodes = config.drop_only_power_nodes
    community_suppression = config.community_suppression
    users_dec_perc_drop = config.users_dec_perc_drop
    items_dec_perc_drop = config.items_dec_perc_drop

    adj_tens = adj_tens.clone()
    device = adj_tens.device

    suppress_mask = torch.zeros(adj_tens.shape[0], dtype=torch.bool, device=adj_tens.device)

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
            suppress_mask[in_com_user_indices[perm]] = True

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
            suppress_mask[in_com_item_indices[perm]] = True  # TODO: handle cases where the dropped edges would overlap

    edge_weights_new = torch.ones(adj_tens.shape[0], dtype=torch.float32, device=device)
    edge_weights_new[suppress_mask] = community_suppression
    return edge_weights_new


def plot_community_confidence(user_probs_path=None, user_labels=None, algorithm='Leiden', force_bipartite=True,
                              save_path='images/', top_n_communities=10, dataset_name=''):
    """
    Create a line plot showing community assignment confidence for each user community.

    :param user_probs_path: Path to pre-saved user rel_rec_freqs CSV, or None to load based on other parameters
    :param user_labels: User community labels tensor, or None to load from save_path
    :param algorithm: Community detection algorithm used ('Leiden' or 'Louvain')
    :param force_bipartite: Whether bipartite structure was enforced
    :param save_path: Directory containing saved files
    :param top_n_communities: Number of top communities to display
    :param dataset_name: Name of the dataset for saving figures
    """

    # Load user rel_rec_freqs and labels if not provided
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

    # Group max rel_rec_freqs by community
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


def initialize_wandb(config):
    """Initialize Weights & Biases for experiment tracking with config object."""
    wandb.login(key="d234bc98a4761bff39de0e5170df00094ac42269")

    # Create wandb config dict from our config object
    wandb_config = {
        "dataset": config.dataset_name,
        "model": config.model_name,
        "users_top_percent": config.users_top_percent,
        "items_top_percent": config.items_top_percent,
        "users_dec_perc_drop": config.users_dec_perc_drop,
        "items_dec_perc_drop": config.items_dec_perc_drop,
        "community_suppression": config.community_suppression,
        "use_dropout": config.use_dropout,
        "drop_only_power_nodes": config.drop_only_power_nodes,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
    }

    # Add model-specific config parameters
    if config.model_name != 'ItemKNN':
        wandb_config.update({
            "train_batch_size": config.train_batch_size,
            "eval_batch_size": config.eval_batch_size,
        })

    if config.model_name == 'LightGCN':
        wandb_config.update({
            "emb_dim": config.embedding_dim,
            "num_layers": config.n_layers,
        })
    elif config.model_name == 'ItemKNN':
        wandb_config.update({
            "item_knn_topk": config.item_knn_topk,
            "shrink": config.shrink,
        })
    elif config.model_name == 'MultiVAE':
        wandb_config.update({
            "hidden_dimension": config.hidden_dimension,
            "latent_dimension": config.latent_dimension,
            "VAE_dropout": config.drop,
            "anneal_cap": config.anneal_cap,
            "total_anneal_steps": config.total_anneal_steps,
        })

    return wandb.init(
        project="RecSys_PowerNodeEdgeDropout",
        name=f"{config.model_name}_{config.dataset_name}_users_top_{config.items_dec_perc_drop}_items_top_{config.items_dec_perc_drop}com_suppression_{config.community_suppression}",
        config=wandb_config
    )


def get_biased_connectivity_data(config, adj_tens):
    user_community_connectivity_matrix, item_community_connectivity_matrix = get_user_item_community_connectivity_matrices(
        adj_tens=adj_tens,
        user_com_labels=config.user_com_labels,
        item_com_labels=config.item_com_labels)

    config.user_community_connectivity_matrix = user_community_connectivity_matrix
    config.item_community_connectivity_matrix = item_community_connectivity_matrix

    config.user_community_connectivity_matrix_distribution = user_community_connectivity_matrix / torch.sum(
        user_community_connectivity_matrix, dim=1, keepdim=True)
    config.item_community_connectivity_matrix_distribution = item_community_connectivity_matrix / torch.sum(
        item_community_connectivity_matrix, dim=1, keepdim=True)

    user_labels_Leiden_matrix_mask = np.loadtxt(f'dataset/{config.dataset_name}/saved/user_labels_matrix_mask.csv',
                                                delimiter=',')
    item_labels_Leiden_matrix_mask = np.loadtxt(f'dataset/{config.dataset_name}/saved/item_labels_matrix_mask.csv',
                                                delimiter=',')

    (config.biased_user_edges_mask,
     config.biased_item_edges_mask) = get_biased_edges_mask(
        config=config,
        adj_tens=adj_tens,
        user_com_labels_mask=torch.tensor(user_labels_Leiden_matrix_mask, device=config.device),
        item_com_labels_mask=torch.tensor(item_labels_Leiden_matrix_mask, device=config.device),
        user_community_connectivity_matrix_distribution=config.user_community_connectivity_matrix_distribution,
        item_community_connectivity_matrix_distribution=config.item_community_connectivity_matrix_distribution,
        )


def get_community_data(config, adj_np):
    """Get or load community labels and power nodes."""
    # Create directory if it doesn't exist
    if not os.path.exists(f'dataset/{config.dataset_name}/saved'):
        os.makedirs(f'dataset/{config.dataset_name}/saved')

    (config.user_com_labels,
     config.item_com_labels) = get_community_labels(
        config=config,
        adj_np=adj_np,
        save_path=f'dataset/{config.dataset_name}/saved',
        get_probs=True)

    config.item_labels_matrix_mask = torch.tensor(np.loadtxt(f'dataset/{config.dataset_name}/saved/item_labels_matrix_mask.csv',
                                                        delimiter=','), dtype=torch.long, device=config.device)
    # config.user_labels_matrix_mask = torch.tensor(np.loadtxt(f'dataset/{config.dataset_name}/saved/user_labels_matrix_mask.csv',
    #                                                     delimiter=','), device=config.device)

    (config.power_users_ids,
     config.power_items_ids) = get_power_users_items(
        adj_tens=torch.tensor(adj_np, device=config.device),
        user_com_labels=config.user_com_labels,
        item_com_labels=config.item_com_labels,
        users_top_percent=config.users_top_percent,
        items_top_percent=config.items_top_percent,
        save_path=f'dataset/{config.dataset_name}/saved')



