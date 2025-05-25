import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.graphgym import train

# from LightGCN_PyTorch.code.register import dataset
# from recbole.data import create_dataset
from LightGCN_PyTorch.code.model import LightGCN
from LightGCN_PyTorch.code.dataloader import Movielens100k
# from RecSys_PyTorch.models import ItemKNN
# from vae_cf_pytorch.models import MultiVAE
from utils_functions import set_seed, plot_community_confidence, plot_community_connectivity_distribution, \
    plot_degree_distributions, plot_connectivity, plot_confidence
from precompute import get_community_connectivity_matrix, get_community_labels, get_power_users_items, \
    get_biased_edges_mask, get_user_item_community_connectivity_matrices
import wandb
from argparse import ArgumentParser
import logging
from logging import getLogger
import numpy as np
from recbole.utils import init_seed
import os
import pandas as pd
from sklearn.model_selection import KFold
from evaluation import evaluate_model, precalculate_average_popularity
from utils_functions import power_node_edge_dropout
from config import Config
from dataset import get_dataset_tensor, LightGCNDataset
from training import train_and_evaluate
import sys
# sys.path.append('/path/to/LightGCN_PyTorch')
# sys.path.append('/path/to/RecSys_PyTorch')


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default='LightGCN')
    parser.add_argument("--dataset_name", type=str, default='ml-100k')
    parser.add_argument("--users_top_percent", type=float, default=0.05)
    parser.add_argument("--items_top_percent", type=float, default=0.05)
    parser.add_argument("--users_dec_perc_drop", type=float, default=0.05)
    parser.add_argument("--items_dec_perc_drop", type=float, default=0.05)
    parser.add_argument("--community_suppression", type=float, default=0.6)
    parser.add_argument("--drop_only_power_nodes", type=bool, default=True)
    parser.add_argument("--use_dropout", type=bool, default=True)
    parser.add_argument("--k_th_fold", type=int, default=0)

    return parser.parse_args()


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
        "k_th_fold": config.k_th_fold,
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
            "emb_dim": config.latent_dim_rec,
            "num_layers": config.lightGCN_n_layers,
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
        name=f"{config.model_name}_{config.dataset_name}_users_top_{config.users_top_percent}_com_drop_strength_{config.community_suppression}",
        config=wandb_config
    )


def get_biased_connectivity_data(config, adj_tens):
    user_community_connectivity_matrix, item_community_connectivity_matrix = calculate_community_metrics(
        config=config,
        adj_tens=adj_tens
    )

    # user/item id 0 is never used / nan as labels start at 1 and we use them for indexing
    # normalize as distribution along the rows
    config.user_community_connectivity_matrix = user_community_connectivity_matrix
    config.item_community_connectivity_matrix = item_community_connectivity_matrix

    user_community_connectivity_matrix[0] = torch.zeros(user_community_connectivity_matrix.shape[1],
                                                        device=config.device)  # node indices start at 1, so we just set a value to be not nan
    item_community_connectivity_matrix[0] = torch.zeros(item_community_connectivity_matrix.shape[1],
                                                        device=config.device)

    config.user_community_connectivity_matrix_distribution = user_community_connectivity_matrix / torch.sum(
        user_community_connectivity_matrix, dim=1, keepdim=True)
    config.item_community_connectivity_matrix_distribution = item_community_connectivity_matrix / torch.sum(
        item_community_connectivity_matrix, dim=1, keepdim=True)



    user_labels_Leiden_matrix_mask = np.loadtxt(f'dataset/{config.dataset_name}/user_labels_Leiden_matrix_mask.csv',
                                                delimiter=',')
    item_labels_Leiden_matrix_mask = np.loadtxt(f'dataset/{config.dataset_name}/item_labels_Leiden_matrix_mask.csv',
                                                delimiter=',')

    (config.biased_user_edges_mask,
     config.biased_item_edges_mask) = get_biased_edges_mask(
        adj_tens=adj_tens,
        user_com_labels_mask=torch.tensor(user_labels_Leiden_matrix_mask, device=config.device),
        item_com_labels_mask=torch.tensor(item_labels_Leiden_matrix_mask, device=config.device),
        user_community_connectivity_matrix_distribution=config.user_community_connectivity_matrix_distribution,
        item_community_connectivity_matrix_distribution=config.item_community_connectivity_matrix_distribution,
        bias_threshold=0.4)


def get_community_data(config, adj_np):
    """Get or load community labels and power nodes."""
    # Create directory if it doesn't exist
    if not os.path.exists(f'dataset/{config.dataset_name}'):
        os.makedirs(f'dataset/{config.dataset_name}')

    # TODO: all community data has to be from whole dataset and masked for the subsets
    (config.user_com_labels,
     config.item_com_labels) = get_community_labels(
        config=config,
        adj_np=adj_np,
        save_path=f'dataset/{config.dataset_name}',
        get_probs=True)

    (config.power_users_ids,
     config.power_items_ids) = get_power_users_items(
        config=config,
        adj_tens=torch.tensor(adj_np, device=config.device),
        user_com_labels=config.user_com_labels,
        item_com_labels=config.item_com_labels,
        users_top_percent=config.users_top_percent,
        items_top_percent=config.items_top_percent,
        save_path=f'dataset/{config.dataset_name}')


def calculate_community_metrics(config, adj_tens):
    """Calculate community connectivity matrix and average degrees."""
    (user_community_connectivity_matrix,
     item_community_connectivity_matrix) = get_user_item_community_connectivity_matrices(
        adj_tens=adj_tens,
        user_com_labels=config.user_com_labels,
        item_com_labels=config.item_com_labels)
    return user_community_connectivity_matrix, item_community_connectivity_matrix


def get_subset_masks(config):
    """Get subset masks for community data."""
    dataset_len = config.train_dataset_len
    fold_size = dataset_len // 5  # 5 folds, test set is excluded
    start = config.k_th_fold * fold_size
    end = (config.k_th_fold + 1) * fold_size if config.k_th_fold !=4 else dataset_len

    valid_mask = np.zeros(dataset_len, dtype=bool)
    valid_mask[start:end] = True
    test_mask = np.zeros(dataset_len, dtype=bool)
    test_mask[fold_size * 5:] = True
    train_mask = np.zeros(dataset_len, dtype=bool)
    train_mask[~valid_mask & ~test_mask] = True

    config.train_mask = train_mask
    config.valid_mask = valid_mask
    config.test_mask = test_mask


def main():
    seed = 42
    set_seed(seed)

    config = Config()

    args = parse_arguments()
    config.update_from_args(args)
    config.setup_model_config()
    config.log_config()

    dataset_tensor = get_dataset_tensor(config)

    test_size = 0.2  # 1 - 0.2 = 0.8 => 0.8 / 5 = 0.16 so whole numbers for 5 folds
    train_dataset_len = int(len(dataset_tensor) * (1 - test_size))  # 80% for training/validation, 20% for testing
    validation_indices = np.arange(config.k_th_fold * (train_dataset_len // 5),
                                   (config.k_th_fold + 1) * (train_dataset_len // 5), dtype=np.int64)

    dataset_all = Movielens100k(validation_indices=validation_indices)
    config.train_dataset_len = train_dataset_len
    config.nr_items = dataset_tensor[:, 1].max()

    # TODO: check community bias and connectivity if correct
    get_community_data(
        config=config,
        adj_np=dataset_tensor.cpu().numpy()
    )

    get_biased_connectivity_data(
        config=config,
        adj_tens=dataset_tensor)

    # plot_connectivity(config.user_community_connectivity_matrix, users_items='users', save_path='images', dataset_name="ml-100k")
    # plot_connectivity(config.item_community_connectivity_matrix, users_items='items', save_path='images', dataset_name='ml-100k')
    # user_probs = np.loadtxt(f'dataset/{config.dataset_name}/user_labels_Leiden_probs.csv', delimiter=',')
    # item_probs = np.loadtxt(f'dataset/{config.dataset_name}/item_labels_Leiden_probs.csv', delimiter=',')
    # plot_confidence(user_probs, save_path='images', dataset_name="ml100k", users_items='users')
    # plot_confidence(item_probs, save_path='images', dataset_name="ml100k", users_items='items')
    # TODO: how to make subset masks in training
    get_subset_masks(config)

    # Initialize model
    if config.model_name == 'LightGCN':
        # TODO: check if I should use dataloader, dataset and model from official LightGCN implementation
        # TODO: use DataLoader and Dataset object from RecSys_pytorch implementation
        model = LightGCN(dataset=dataset_all, config=vars(config)).to(config.device)
    elif config.model_name == 'ItemKNN':
        model = ItemKNN(n_users=train_dataset.n_users, n_items=train_dataset.n_items).to(config.device)
    elif config.model_name == 'MultVAE':
        model = MultiVAE(dataset=train_dataset, hyperparams=config, device=config.device).to(config.device)
    else:
        raise ValueError(f"Model {config.model_name} not supported")

    wandb_run = initialize_wandb(config)

    # Train and evaluate
    best_valid_score, test_metrics = train_and_evaluate(
        config=config,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )

    # Finalize wandb
    wandb.log({"final_test_metrics": test_metrics})
    rng_id = np.random.randint(0, 100000)
    wandb.save(f"{config.model_name}_{config.dataset_name}_ID{rng_id}.pth")
    wandb_run.finish()


if __name__ == "__main__":
    main()


