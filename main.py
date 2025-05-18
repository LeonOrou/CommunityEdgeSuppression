import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models import LightGCN as PT_LightGCN
from models import ItemKNN as PT_ItemKNN
from models import MultiVAE as PT_MultiVAE
from utils_functions import set_seed, plot_community_confidence, plot_community_connectivity_distribution, plot_degree_distributions
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


class InteractionDataset(Dataset):
    def __init__(self, df):
        self.df = df  # store DataFrame for later dropout updates
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float)

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
    parser.add_argument("--community_dropout_strength", type=float, default=0.6)
    parser.add_argument("--drop_only_power_nodes", type=bool, default=True)
    parser.add_argument("--use_dropout", type=bool, default=True)
    parser.add_argument("--k_th_fold", type=int, default=0)

    return parser.parse_args()


def load_config_from_yaml(dataset_name):
    """Load configuration from YAML file."""
    with open(f'{dataset_name}_config.yaml', 'r') as file:
        config_file = yaml.safe_load(file)
        return {
            'rating_col_name': config_file['RATING_FIELD'],
            'topk': config_file['topk'],
            'learning_rate': config_file['learning_rate'],
        }


def setup_device(try_gpu=True):
    """Setup device (CPU/GPU)."""
    if try_gpu:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device('cpu')


def setup_config(args, device, seed):
    """Setup configuration."""
    config_dict = {
        'users_dec_perc_drop': args.users_dec_perc_drop,
        'items_dec_perc_drop': args.items_dec_perc_drop,
        'community_dropout_strength': args.community_dropout_strength,
        'drop_only_power_nodes': args.drop_only_power_nodes,
        'patience': 2,
        'gamma': 0.5,
        'min_lr': 1e-5,
        'scheduler': 'plateau',
        'reproducibility': True,
        'device': device,
        'k_th_fold': args.k_th_fold,
    }

    if args.model_name == 'LightGCN':
        config_dict['train_batch_size'] = 256
        config_dict['eval_batch_size'] = 256
        config_dict['epochs'] = 200
        config_dict['n_layers'] = 5  # from model hyperparameter search
        config_dict['embedding_size'] = 256
    elif args.model_name == 'ItemKNN':
        config_dict['epochs'] = 1
        config_dict['k'] = 250
        config_dict['shrink'] = 10  # from model hyperparameter search
    elif args.model_name == 'MultiVAE':
        config_dict['epochs'] = 200
        config_dict['train_batch_size'] = 4096
        config_dict['eval_batch_size'] = 4096
        config_dict['hidden_dimension'] = 800  # from model hyperparameter search
        config_dict['latent_dimension'] = 200
        config_dict['dropout_prob'] = 0.7
        config_dict['anneal_cap'] = 0.3
        config_dict['total_anneal_steps'] = 200000

    config = config_dict
    init_seed(seed=seed, reproducibility=True)
    logger = getLogger()
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)
    logger.info(config)

    return config, logger


def initialize_wandb(args, config_params, config):
    """Initialize Weights & Biases for experiment tracking."""
    wandb.login(key="d234bc98a4761bff39de0e5170df00094ac42269")
    return wandb.init(
        project="RecSys_PowerNodeEdgeDropout",
        name=f"{args.model_name}_{args.dataset_name}_users_top_{args.users_top_percent}_com_drop_strength_{args.community_dropout_strength}",
        config={
            "dataset": args.dataset_name,
            "model": args.model_name,
            "users_top_percent": args.users_top_percent,
            "items_top_percent": args.items_top_percent,
            "users_dec_perc_drop": args.users_dec_perc_drop,
            "items_dec_perc_drop": args.items_dec_perc_drop,
            "community_dropout_strength": args.community_dropout_strength,
            "use_dropout": args.use_dropout,
            "drop_only_power_nodes": args.drop_only_power_nodes,
            "k_th_fold": args.k_th_fold,
            "TopK": config_params['topk'],
            "learning_rate": config_params['learning_rate'],
            "epochs": config['epochs'],
            "train_batch_size": config['train_batch_size'] if args.model_name != 'ItemKNN' else None,
            "eval_batch_size": config['eval_batch_size'] if args.model_name != 'ItemKNN' else None,
            "embedding_size": config['embedding_size'] if args.model_name == 'LightGCN' else None,
            "n_layers": config['n_layers'] if args.model_name == 'LightGCN' else None,
            "k_ItemKNN": config['k'] if args.model_name == 'ItemKNN' else None,
            "shrink": config['shrink'] if args.model_name == 'ItemKNN' else None,
            "hidden_dimension": config['hidden_dimension'] if args.model_name == 'MultiVAE' else None,
            "latent_dimension": config['latent_dimension'] if args.model_name == 'MultiVAE' else None,
            "dropout_prob": config['dropout_prob'] if args.model_name == 'MultiVAE' else None,
            "anneal_cap": config['anneal_cap'] if args.model_name == 'MultiVAE' else None,
            "total_anneal_steps": config['total_anneal_steps'] if args.model_name == 'MultiVAE' else None,
        }
    )


def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for users, items, ratings in data_loader:
            users = users.to(device)
            items = items.to(device)
            outputs = model(users, items)
            all_preds.extend(outputs.cpu().numpy().tolist())
            all_labels.extend(ratings.cpu().numpy().tolist())
    metrics = {'ndcg': 0, 'recall': 0, 'hit': 0, 'avg_popularity': 0, 'gini': 0, 'coverage': 0}
    return metrics

    return best_valid_score, best_valid_result, trainer


def get_biased_connectivity_data(config, adj_tens):
    device = config['device']
    user_community_connectivity_matrix, item_community_connectivity_matrix = calculate_community_metrics(config=config,
                                                                                                         adj_tens=adj_tens,
                                                                                                         device=device)
    # user/item id 0 is never used / nan as labels start at 1 and we use them for indexing
    # normalize as distribution along the rows
    user_community_connectivity_matrix = user_community_connectivity_matrix / torch.sum(user_community_connectivity_matrix, dim=1, keepdim=True)
    item_community_connectivity_matrix = item_community_connectivity_matrix / torch.sum(item_community_connectivity_matrix, dim=1, keepdim=True)

    config['user_community_connectivity_matrix'] = user_community_connectivity_matrix
    config['item_community_connectivity_matrix'] = item_community_connectivity_matrix

    user_community_connectivity_matrix[0] = torch.zeros(user_community_connectivity_matrix.shape[1], device=device)  # node indices start at 1, so we just set a value to be not nan
    item_community_connectivity_matrix[0] = torch.zeros(item_community_connectivity_matrix.shape[1], device=device)
    config['user_community_connectivity_matrix_distribution'] = user_community_connectivity_matrix
    config['item_community_connectivity_matrix_distribution'] = item_community_connectivity_matrix

    user_labels_Leiden_matrix_mask = np.loadtxt(f'dataset/{config.dataset}/user_labels_Leiden_matrix_mask.csv', delimiter=',')
    item_labels_Leiden_matrix_mask = np.loadtxt(f'dataset/{config.dataset}/item_labels_Leiden_matrix_mask.csv', delimiter=',')

    (config['biased_user_edges_mask'],
     config['biased_item_edges_mask']) = get_biased_edges_mask(
        adj_tens=adj_tens,
        user_com_labels_mask=torch.tensor(user_labels_Leiden_matrix_mask, device=device),
        item_com_labels_mask=torch.tensor(item_labels_Leiden_matrix_mask, device=device),
        user_community_connectivity_matrix_distribution=user_community_connectivity_matrix,
        item_community_connectivity_matrix_distribution=item_community_connectivity_matrix,
        bias_threshold=0.4)


def get_community_data(config, adj_np, device, users_top_percent, items_top_percent):
    """Get or load community labels and power nodes."""
    # Create directory if it doesn't exist
    if not os.path.exists(f'dataset/{config.dataset}'):
        os.makedirs(f'dataset/{config.dataset}')

    # TODO: all community data has to be from whole dataset and masked for the subsets
    (config['user_com_labels'],
     config['item_com_labels']) = get_community_labels(
        config=config,
        adj_np=adj_np,
        save_path=f'dataset/{config.dataset}',
        get_probs=True)

    (config['power_users_ids'],
     config['power_items_ids']) = get_power_users_items(
        config=config,
        adj_tens=torch.tensor(adj_np, device=device),
        user_com_labels=config.variable_config_dict['user_com_labels'],
        item_com_labels=config.variable_config_dict['item_com_labels'],
        users_top_percent=users_top_percent,
        items_top_percent=items_top_percent,
        save_path=f'dataset/{config.dataset}')


def calculate_community_metrics(config, adj_tens, device):
    """Calculate community connectivity matrix and average degrees."""

    (user_community_connectivity_matrix,
     item_community_connectivity_matrix) = get_user_item_community_connectivity_matrices(adj_tens=adj_tens,
                                                                                      user_com_labels=config.variable_config_dict['user_com_labels'],
                                                                                      item_com_labels=config.variable_config_dict['item_com_labels'])
    return user_community_connectivity_matrix, item_community_connectivity_matrix


def get_subset_masks(config, k_th_fold):
    """Get subset masks for community data."""
    dataset_len = config.variable_config_dict['dataset_len']
    fold_size = dataset_len / 6  # has to be int, calculated in create_k_folded_local_dataset!
    start = k_th_fold * fold_size
    end = (k_th_fold + 1) * fold_size if k_th_fold != 5 else dataset_len

    valid_mask = np.zeros(dataset_len, dtype=bool)
    valid_mask[start:end] = True
    test_mask = np.zeros(dataset_len, dtype=bool)
    test_mask[fold_size * 5:] = True
    train_mask = np.zeros(dataset_len, dtype=bool)
    train_mask[~valid_mask & ~test_mask] = True

    config['train_mask'] = train_mask
    config['valid_mask'] = valid_mask
    config['test_mask'] = test_mask


def train_and_evaluate(config, model, train_dataset, test_dataset, device, args):
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    epochs = config['epochs']
    best_valid_score = -float('inf')

    kf = KFold(n_splits=5, shuffle=True)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_dataset)):
        print(f"Fold {fold + 1}")
        print("-------")

        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config['batch_size'],
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        valid_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config['batch_size'],
            sampler=torch.utils.data.SubsetRandomSampler(valid_idx),
        )
        for epoch in range(epochs):
            model.train()
            from utils_functions import power_node_edge_dropout
            train_df = train_loader.dataset.df.copy()
            train_df = power_node_edge_dropout(train_df, args)
            train_loader.dataset = InteractionDataset(train_df)

            epoch_loss = 0.0
            for users, items, ratings in train_loader:
                users = users.to(device)
                items = items.to(device)
                ratings = ratings.to(device)
                optimizer.zero_grad()
                outputs = model(users, items)
                loss = criterion(outputs.squeeze(), ratings)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            metrics = evaluate_model(model, valid_loader, device)
            if metrics['ndcg'] > best_valid_score:
                best_valid_score = metrics['ndcg']
                torch.save(model.state_dict(), f"{args.model_name}_{args.dataset_name}_best.pth")

            wandb.log({"epoch": epoch, "loss": epoch_loss, **metrics})
    # test the finished model
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config['batch_size'],
        sampler=...,
    )
    test_metrics = evaluate_model(model, test_loader, device)
    return best_valid_score, test_metrics


def get_dataset_tensor(dataset_name, device):
    interaction = np.loadtxt(f'dataset/{dataset_name}/{dataset_name}.inter', delimiter=' ', skiprows=1)
    interaction = interaction[:, :3]  # get only user_id, item_id, rating columns
    # if not all are 1, i.e. not processed yet
    if np.all(interaction[:, 2] != 1):
        # if all are 1, we need to binarize the ratings
        interaction = interaction[interaction[:, 2] >= 4]  # get only ratings with 4 and above
        interaction[:, 2] = 1  # binarize the ratings
        # get only nodes with a degree of at least 10
        user_degrees = np.bincount(interaction[:, 0].astype(int))
        item_degrees = np.bincount(interaction[:, 1].astype(int))
        valid_users = np.where(user_degrees >= 10)[0]
        valid_items = np.where(item_degrees >= 10)[0]
        interaction = interaction[np.isin(interaction[:, 0], valid_users) & np.isin(interaction[:, 1], valid_items)]

    np.random.shuffle(interaction)

    return torch.tensor(interaction, dtype=torch.int64, device=device)


def main():
    seed = 42
    set_seed(seed)
    args = parse_arguments()
    config_params = load_config_from_yaml(args.dataset_name)
    device = setup_device(try_gpu=True)
    config, logger = setup_config(args, device, seed)
    config = dict(config)
    config['learning_rate'] = config_params.get('learning_rate', 0.001)

    dataset_tensor = get_dataset_tensor(args.dataset_name, device)
    train_dataset = InteractionDataset(dataset_tensor[:int(len(dataset_tensor) * 0.8)])
    test_dataset = InteractionDataset(dataset_tensor[int(len(dataset_tensor) * 0.8):])

    wandb_run = initialize_wandb(args, config_params, config)

    get_community_data(
        config=config,
        adj_np=dataset_tensor.cpu().numpy(),
        device=device,
        users_top_percent=args.users_top_percent,
        items_top_percent=args.items_top_percent
    )
    get_biased_connectivity_data(config=config, adj_tens=torch.tensor(dataset_tensor.cpu().numpy(), device=device))
    get_subset_masks(config=config, k_th_fold=args.k_th_fold)

    if args.model_name == 'LightGCN':
        model = PT_LightGCN(config).to(device)
    elif args.model_name == 'ItemKNN':
        model = PT_ItemKNN(config).to(device)
    elif args.model_name == 'MultiVAE':
        model = PT_MultiVAE(config).to(device)
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    best_valid_score, test_metrics = train_and_evaluate(config=config,
                                                        model=model,
                                                        train_dataset=train_dataset,
                                                        test_dataset=test_dataset,
                                                        device=device,
                                                        args=args)
    wandb.log({"final_test_metrics": test_metrics})
    rng_id = np.random.randint(0, 100000)
    wandb.save(f"{args.model_name}_{args.dataset_name}_ID{rng_id}.pth")
    wandb_run.finish()


if __name__ == "__main__":
    main()
