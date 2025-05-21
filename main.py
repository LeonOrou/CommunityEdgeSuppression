import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.graphgym import train

# from LightGCN_PyTorch.code.register import dataset
from recbole.data import create_dataset
from LightGCN_PyTorch.code.model import LightGCN
from RecSys_PyTorch.models import ItemKNN
from vae_cf_pytorch.models import MultiVAE
from utils_functions import set_seed, plot_community_confidence, plot_community_connectivity_distribution, \
    plot_degree_distributions, plot_connectivity, plot_confidence
from precompute import get_community_connectivity_matrix, get_community_labels, get_power_users_items, \
    get_biased_edges_mask, get_user_item_community_connectivity_matrices
import wandb
from argparse import ArgumentParser
import yaml
import logging
from logging import getLogger
import numpy as np
from recbole.utils import init_seed
import os
import pandas as pd
from sklearn.model_selection import KFold
from evaluation import evaluate_model, precalculate_average_popularity
from utils_functions import power_node_edge_dropout
import sys
# sys.path.append('/path/to/LightGCN_PyTorch')
# sys.path.append('/path/to/RecSys_PyTorch')


class Config:
    """Central configuration class to store all parameters."""

    def __init__(self):
        self.model_name = None
        self.dataset_name = None
        self.users_top_percent = None
        self.items_top_percent = None
        self.users_dec_perc_drop = None
        self.items_dec_perc_drop = None
        self.community_suppression = None
        self.drop_only_power_nodes = None
        self.use_dropout = None
        self.k_th_fold = None

        # Will be set during initialization
        self.device = None
        self.user_com_labels = None
        self.item_com_labels = None
        self.power_users_ids = None
        self.power_items_ids = None
        self.user_community_connectivity_matrix = None
        self.item_community_connectivity_matrix = None
        self.user_community_connectivity_matrix_distribution = None
        self.item_community_connectivity_matrix_distribution = None
        self.biased_user_edges_mask = None
        self.biased_item_edges_mask = None
        self.train_mask = None
        self.valid_mask = None
        self.test_mask = None
        self.train_dataset_len = 0

        # training parameters; lr-scheduler, optimizer, etc.
        self.patience = 2
        self.gamma = 0.5
        self.min_lr = 1e-5
        self.reproducibility = True
        self.learning_rate = 1e-3
        self.nr_items = None

        self.setup_device()

    def update_from_args(self, args):
        """Update config from command line arguments."""
        for key, value in vars(args).items():
            setattr(self, key, value)

    def setup_model_config(self):
        """Setup model-specific configurations."""
        if self.model_name == 'LightGCN':
            self.train_batch_size = 512
            self.eval_batch_size = 512
            self.batch_size = 512  # For consistency
            self.epochs = 200  # because it's different for each model
            self.num_layers = 5
            self.emb_dim = 128
            self.split = False
            self.num_folds = 5
            self.node_dropout = 0.0
            self.reg = 1e-4
            self.graph_dir = f'./dataset/{self.dataset_name}/lgcn_graphs'
        elif self.model_name == 'ItemKNN':
            self.epochs = 1
            self.item_knn_topk = 250
            self.shrink = 10
            self.feature_weighting = 'bm25'
        elif self.model_name == 'MultiVAE':
            self.epochs = 200
            self.train_batch_size = 4096
            self.eval_batch_size = 4096
            self.batch_size = 4096  # For consistency
            self.hidden_dimension = 800
            self.latent_dimension = 200
            self.q_dims = [self.hidden_dimension, self.latent_dimension]
            self.p_dims = [self.latent_dimension, self.hidden_dimension, self.nr_items]
            self.drop = 0.7
            self.anneal_cap = 0.3
            self.total_anneal_steps = 200000

    def setup_device(self, try_gpu=True):
        """Setup computation device."""
        if try_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        return self.device

    def log_config(self):
        """Log current configuration."""
        logger = getLogger()
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        logger.addHandler(c_handler)
        logger.info(vars(self))
        return logger


class InteractionDataset(Dataset):
    def __init__(self, adj):
        self.adj = adj  # store DataFrame for later dropout updates
        self.users = torch.tensor(adj[:, 0], dtype=torch.int32)
        self.items = torch.tensor(adj[:, 1], dtype=torch.int32)
        self.ratings = torch.tensor(adj[:, 2], dtype=torch.int8)

        self.n_users = int(adj[:, 0].max()) + 1
        self.n_items = int(adj[:, 1].max()) + 1

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


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
            "emb_dim": config.emb_dim,
            "num_layers": config.num_layers,
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


def evaluate_model(model, data_loader, config):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for users, items, ratings in data_loader:
            users = users.to(config.device)
            items = items.to(config.device)
            outputs = model(users, items)
            all_preds.extend(outputs.cpu().numpy().tolist())
            all_labels.extend(ratings.cpu().numpy().tolist())

    # TODO: call evaluate_model and pass metrics to k_th_folded dict
    metrics = {'ndcg': 0, 'recall': 0, 'hit': 0, 'avg_popularity': 0, 'gini': 0, 'coverage': 0}

    return metrics


def get_biased_connectivity_data(config, adj_tens):
    user_community_connectivity_matrix, item_community_connectivity_matrix = calculate_community_metrics(
        config=config,
        adj_tens=adj_tens
    )

    # user/item id 0 is never used / nan as labels start at 1 and we use them for indexing
    # normalize as distribution along the rows
    config.user_community_connectivity_matrix = user_community_connectivity_matrix
    config.item_community_connectivity_matrix = item_community_connectivity_matrix

    config.user_community_connectivity_matrix_distribution = user_community_connectivity_matrix / torch.sum(
        user_community_connectivity_matrix, dim=1, keepdim=True)
    config.item_community_connectivity_matrix_distribution = item_community_connectivity_matrix / torch.sum(
        item_community_connectivity_matrix, dim=1, keepdim=True)

    user_community_connectivity_matrix[0] = torch.zeros(user_community_connectivity_matrix.shape[1],
                                                        device=config.device)  # node indices start at 1, so we just set a value to be not nan
    item_community_connectivity_matrix[0] = torch.zeros(item_community_connectivity_matrix.shape[1],
                                                        device=config.device)

    user_labels_Leiden_matrix_mask = np.loadtxt(f'dataset/{config.dataset_name}/user_labels_Leiden_matrix_mask.csv',
                                                delimiter=',')
    item_labels_Leiden_matrix_mask = np.loadtxt(f'dataset/{config.dataset_name}/item_labels_Leiden_matrix_mask.csv',
                                                delimiter=',')

    (config.biased_user_edges_mask,
     config.biased_item_edges_mask) = get_biased_edges_mask(
        adj_tens=adj_tens,
        user_com_labels_mask=torch.tensor(user_labels_Leiden_matrix_mask, device=config.device),
        item_com_labels_mask=torch.tensor(item_labels_Leiden_matrix_mask, device=config.device),
        user_community_connectivity_matrix_distribution=user_community_connectivity_matrix,
        item_community_connectivity_matrix_distribution=item_community_connectivity_matrix,
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


def get_dataset_tensor(config):
    """Get dataset tensor using config object."""
    if not os.path.exists(f'dataset/{config.dataset_name}/{config.dataset_name}_processed.inter'):
        min_degree = 5 if 'ml' in config.dataset_name else 5  # 10 for lfm?
        min_rating = 4 if 'ml' in config.dataset_name else 5  # in lfm rating is number of listening event
        interaction = np.loadtxt(f'dataset/{config.dataset_name}/{config.dataset_name}.inter', delimiter=' ', skiprows=1)
        interaction = interaction[:, :3]  # get only user_id, item_id, rating columns
        # if all are 1, we need to binarize the ratings
        interaction = interaction[interaction[:, 2] >= min_rating]  # get only ratings with 4 and above
        interaction[:, 2] = 1  # binarize the ratings
        user_degrees = np.bincount(interaction[:, 0].astype(int))
        item_degrees = np.bincount(interaction[:, 1].astype(int))
        valid_users = np.where(user_degrees >= min_degree)[0]
        valid_items = np.where(item_degrees >= min_degree)[0]
        interaction = interaction[np.isin(interaction[:, 0], valid_users) & np.isin(interaction[:, 1], valid_items)]
        np.random.shuffle(interaction)
        np.savetxt(f'dataset/{config.dataset_name}/{config.dataset_name}_processed.inter', interaction, fmt='%d', delimiter=' ')
    else:
        interaction = np.loadtxt(f'dataset/{config.dataset_name}/{config.dataset_name}_processed.inter', delimiter=' ', skiprows=1)

    config.train_dataset_len = len(interaction)

    return torch.tensor(interaction, dtype=torch.int32, device=config.device)


def train_and_evaluate(config, model, train_dataset, test_dataset):
    """Train and evaluate model using config for all parameters"""
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # Create a learning rate scheduler that reduces LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # max ndcg
        factor=config.gamma,
        patience=config.patience,
        min_lr=config.min_lr,
        verbose=True
    )

    criterion = nn.BCEWithLogitsLoss()
    best_valid_score = -float('inf')

    kf = KFold(n_splits=5, shuffle=True)
    results = {}
    # calculate average popularity dict
    avg_item_pop = precalculate_average_popularity(train_dataset.adj)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_dataset)):
        print(f"Fold {fold + 1}")
        print("-------")

        valid_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(valid_idx),
        )

        for epoch in range(config.epochs):
            model.train()
            train_adj_i = power_node_edge_dropout(
                train_dataset.adj,
                power_users_idx=config.power_users_ids,
                power_items_idx=config.power_items_ids,
                biased_user_edges_mask=config.biased_user_edges_mask,
                biased_item_edges_mask=config.biased_item_edges_mask,
                drop_only_power_nodes=config.drop_only_power_nodes,
                community_suppression=config.community_suppression,
                users_dec_perc_drop=config.users_dec_perc_drop,
                items_dec_perc_drop=config.items_dec_perc_drop
            )
            train_dataset_epoch_i = InteractionDataset(train_adj_i)

            # we make the loader after the dropout to not edit the loader
            train_loader = DataLoader(
                dataset=train_dataset_epoch_i,
                batch_size=config.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(train_idx),
                pin_memory=True,  # for faster data transfer to GPU
            )

            epoch_loss = 0.0
            for users, items, ratings in train_loader:
                users = users.to(config.device)
                items = items.to(config.device)
                ratings = ratings.to(config.device)
                optimizer.zero_grad()
                outputs = model(users, items)
                loss = criterion(outputs.squeeze(), ratings)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            metrics_eval = evaluate_model(model=model, test_loader=valid_loader, device=config.device, item_popularity=avg_item_pop)
            results[fold] = metrics_eval
            # Update learning rate scheduler based on the validation metric
            scheduler.step(metrics_eval[10]['ndcg'])

            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"learning_rate": current_lr})

            if metrics_eval[10]['ndcg'] > best_valid_score:  # eval at ndcg@10
                best_valid_score = metrics_eval['ndcg']
                torch.save(model.state_dict(), f"{config.model_name}_{config.dataset_name}_best.pth")

            wandb.log({"epoch": epoch, "loss": epoch_loss, **metrics_eval})

    results_folds_averages_each_k = {
        k: {metric: sum(fold[k][metric] for fold in results.values()) / len(results) for metric in
            next(iter(results.values()))[k].keys()} for k in next(iter(results.values())).keys()}

    # test the finished model
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_metrics = evaluate_model(model, test_loader, config)
    return best_valid_score, test_metrics


def main():
    seed = 42
    set_seed(seed)

    config = Config()

    args = parse_arguments()
    config.update_from_args(args)
    config.setup_model_config()
    config.log_config()

    recbole_dataset = create_dataset(config.dataset_name)

    dataset_tensor = get_dataset_tensor(config)
    test_size = 0.2  # 1 - 0.2 = 0.8 => 0.8 / 5 = 0.16 so whole numbers for 5 folds
    train_dataset = InteractionDataset(dataset_tensor[:int(len(dataset_tensor) * (1 - test_size))])
    test_dataset = InteractionDataset(dataset_tensor[int(len(dataset_tensor) * (1 - test_size)):])
    config.train_dataset_len = len(train_dataset)
    config.nr_items = dataset_tensor[:, 1].max()

    # TODO: check community bias and connectivity if correct
    get_community_data(
        config=config,
        adj_np=dataset_tensor.cpu().numpy()
    )

    get_biased_connectivity_data(
        config=config,
        adj_tens=dataset_tensor, device=config.device)

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
        model = LightGCN(dataset=train_dataset, hyperparams=config, device=config.device).to(config.device)
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


