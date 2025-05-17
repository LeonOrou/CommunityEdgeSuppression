from recbole.data.dataloader import TrainDataLoader
from recbole.data.interaction import Interaction
from recbole.sampler import RepeatableSampler

from utils_functions import set_seed, plot_community_confidence, plot_community_connectivity_distribution, plot_degree_distributions
from precompute import get_community_connectivity_matrix, get_community_labels, get_power_users_items, \
    get_biased_edges_mask, get_user_item_community_connectivity_matrices
from recbole.data import create_dataset, data_preparation
import wandb
from argparse import ArgumentParser
import yaml
import logging
from logging import getLogger
from recbole.utils import init_logger
from recbole.config import Config
from recbole.model.general_recommender import LightGCN, ItemKNN, MultiVAE
import numpy as np
from recbole.utils import init_seed
import torch
import copy
from PowerDropoutTrainer import PowerDropoutTrainer
from recbole.trainer import Trainer
import os
import pandas as pd


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser()
    # in cmd: python main.py --model_name LightGCN --dataset_name ml-20m --config_file_name ml-20_config.yaml --users_top_percent 0.01 --users_dec_perc_drop 0.70 --community_dropout_strength 0.5 --do_power_nodes_from_community True
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
    """Setup RecBole configuration."""
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

    config = Config(
        model=args.model_name,
        dataset=args.dataset_name,
        config_file_list=[f'{args.dataset_name}_config.yaml'],
        config_dict=config_dict
    )
    config['device'] = device

    # Initialize seed and logging
    init_seed(seed=seed, reproducibility=True)
    init_logger(config)
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
            "epochs": config.variable_config_dict['epochs'],
            "train_batch_size": config.variable_config_dict['train_batch_size'] if args.model_name != 'ItemKNN' else None,
            "eval_batch_size": config.variable_config_dict['eval_batch_size'] if args.model_name != 'ItemKNN' else None,
            "embedding_size": config.variable_config_dict['embedding_size'] if args.model_name == 'LightGCN' else None,
            "n_layers": config.variable_config_dict['n_layers'] if args.model_name == 'LightGCN' else None,
            "k_ItemKNN": config.variable_config_dict['k'] if args.model_name == 'ItemKNN' else None,
            "shrink": config.variable_config_dict['shrink'] if args.model_name == 'ItemKNN' else None,
            "hidden_dimension": config.variable_config_dict['hidden_dimension'] if args.model_name == 'MultiVAE' else None,
            "latent_dimension": config.variable_config_dict['latent_dimension'] if args.model_name == 'MultiVAE' else None,
            "dropout_prob": config.variable_config_dict['dropout_prob'] if args.model_name == 'MultiVAE' else None,
            "anneal_cap": config.variable_config_dict['anneal_cap'] if args.model_name == 'MultiVAE' else None,
            "total_anneal_steps": config.variable_config_dict['total_anneal_steps'] if args.model_name == 'MultiVAE' else None,
        }
    )


def get_adj_from_object(data_loader_object, device):
    """Preprocess training data to adjacency matrix format."""
    data_coo = copy.deepcopy(data_loader_object.dataset).inter_matrix()
    indices = torch.tensor((np.concatenate((data_coo.row, data_coo.col), axis=1)), dtype=torch.int32, device=device).T
    values = torch.unsqueeze(torch.tensor(data_coo.data, dtype=torch.int32, device=device), dim=0).T
    adj_np = np.array(torch.cat((indices, values), dim=1).cpu(), dtype=np.int64)
    return adj_np


def get_community_data(config, adj_np, device, users_top_percent, items_top_percent):
    """Get or load community labels and power nodes."""
    # Create directory if it doesn't exist
    if not os.path.exists(f'dataset/{config.dataset}'):
        os.makedirs(f'dataset/{config.dataset}')

    # TODO: all community data has to be from whole dataset and masked for the subsets
    (config.variable_config_dict['user_com_labels'],
     config.variable_config_dict['item_com_labels']) = get_community_labels(
        config=config,
        adj_np=adj_np,
        save_path=f'dataset/{config.dataset}',
        get_probs=True)

    (config.variable_config_dict['power_users_ids'],
     config.variable_config_dict['power_items_ids']) = get_power_users_items(
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


def initialize_model(model_name, config, train_data):
    """Initialize the recommendation model."""
    if model_name == 'LightGCN':
        model = LightGCN(config, train_data.dataset).to(config['device'])
    elif model_name == 'ItemKNN':
        model = ItemKNN(config, train_data.dataset).to(config['device'])
    elif model_name == 'MultiVAE':
        model = MultiVAE(config, train_data.dataset).to(config['device'])
    else:
        raise ValueError(f"Model {model_name} not supported")

    logger = getLogger()
    logger.info(model)
    return model


def train_and_evaluate(config, model, train_data, valid_data, test_data, use_dropout=True):
    """Train and evaluate the model."""
    if use_dropout:
        trainer = PowerDropoutTrainer(config, model)
    else:
        trainer = Trainer(config, model)

    best_valid_score, best_valid_result = trainer.fit(train_data, test_data, saved=True)

    wandb.log({"best_valid_score": best_valid_score, "best_valid_result": best_valid_result})
    logger = getLogger()
    logger.info(f"Best valid score: {best_valid_score}, best valid result: {best_valid_result}")

    # Evaluate on validation data
    eval_result = trainer.evaluate(valid_data)

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

    config.variable_config_dict['user_community_connectivity_matrix'] = user_community_connectivity_matrix
    config.variable_config_dict['item_community_connectivity_matrix'] = item_community_connectivity_matrix

    user_community_connectivity_matrix[0] = torch.zeros(user_community_connectivity_matrix.shape[1], device=device)  # node indices start at 1, so we just set a value to be not nan
    item_community_connectivity_matrix[0] = torch.zeros(item_community_connectivity_matrix.shape[1], device=device)
    config.variable_config_dict['user_community_connectivity_matrix_distribution'] = user_community_connectivity_matrix
    config.variable_config_dict['item_community_connectivity_matrix_distribution'] = item_community_connectivity_matrix

    user_labels_Leiden_matrix_mask = np.loadtxt(f'dataset/{config.dataset}/user_labels_Leiden_matrix_mask.csv', delimiter=',')
    item_labels_Leiden_matrix_mask = np.loadtxt(f'dataset/{config.dataset}/item_labels_Leiden_matrix_mask.csv', delimiter=',')

    (config.variable_config_dict['biased_user_edges_mask'],
     config.variable_config_dict['biased_item_edges_mask']) = get_biased_edges_mask(
        adj_tens=adj_tens,
        user_com_labels_mask=torch.tensor(user_labels_Leiden_matrix_mask, device=device),
        item_com_labels_mask=torch.tensor(item_labels_Leiden_matrix_mask, device=device),
        user_community_connectivity_matrix_distribution=user_community_connectivity_matrix,
        item_community_connectivity_matrix_distribution=item_community_connectivity_matrix,
        bias_threshold=0.4)


def create_k_folded_local_dataset(k=6, dataset='ml-100k'):
    interaction = np.loadtxt(f'dataset/{dataset}/{dataset}.inter', delimiter=' ', skiprows=1)
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
    # split into k folds
    fold_size = len(interaction) // k
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else len(interaction)
        fold = interaction[start:end]
        np.savetxt(f'dataset/{dataset}/{dataset}_fold_{i}.csv', fold, fmt='%d')


def get_datasets(config, args):
    """Prepare dataset using a specific fold for validation."""
    if not os.path.exists(f'dataset/{args.dataset_name}/{args.dataset_name}_fold_0.csv'):
        create_k_folded_local_dataset(k=6, dataset=args.dataset_name)

    # Load the original dataset
    original_dataset = create_dataset(config)
    logger = getLogger()
    logger.info(original_dataset)

    # Load fold data
    train_folds = []
    for i in range(6):  # Assuming 6 folds
        if i != args.k_th_fold and i != 5:  # Fold 5 reserved for testing
            fold_data = np.loadtxt(f'dataset/{args.dataset_name}/{args.dataset_name}_fold_{i}.csv')
            train_folds.extend(fold_data.tolist())

    valid_fold = np.loadtxt(f'dataset/{args.dataset_name}/{args.dataset_name}_fold_{args.k_th_fold}.csv')
    test_fold = np.loadtxt(
        f'dataset/{args.dataset_name}/{args.dataset_name}_fold_5.csv')  # Always use fold 5 for testing

    # Convert to DataFrames
    train_df = pd.DataFrame(train_folds, columns=['user_id', 'item_id', 'rating'])
    valid_df = pd.DataFrame(valid_fold, columns=['user_id', 'item_id', 'rating'])
    test_df = pd.DataFrame(test_fold, columns=['user_id', 'item_id', 'rating'])

    # Create custom dataset with the fold data
    train_dataset = original_dataset.copy(new_inter_feat=Interaction(train_df))

    # Create data loaders
    train_data = TrainDataLoader(
        config=config,
        dataset=train_dataset,
        sampler=RepeatableSampler(phases='train', dataset=train_dataset),
        shuffle=config['shuffle']
    )

    valid_dataset = original_dataset.copy(new_inter_feat=Interaction(valid_df))
    valid_data = TrainDataLoader(
        config=config,
        dataset=valid_dataset,
        sampler=RepeatableSampler(phases='valid', dataset=valid_dataset),
        shuffle=False
    )

    test_dataset = original_dataset.copy(new_inter_feat=Interaction(test_df))
    test_data = TrainDataLoader(
        config=config,
        dataset=test_dataset,
        sampler=RepeatableSampler(phases='test', dataset=test_dataset),
        shuffle=False
    )

    config.variable_config_dict['dataset_len'] = len(train_folds) + len(valid_fold) + len(test_fold)

    return train_data, valid_data, test_data


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

    config.variable_config_dict['train_mask'] = train_mask
    config.variable_config_dict['valid_mask'] = valid_mask
    config.variable_config_dict['test_mask'] = test_mask





def main():
    # Set seed for reproducibility
    seed = 42
    set_seed(seed)

    args = parse_arguments()

    config_params = load_config_from_yaml(args.dataset_name)

    device = setup_device(try_gpu=True)

    config, logger = setup_config(args, device, seed)

    train_data, valid_data, test_data = get_datasets(config=config, args=args)

    wandb_run = initialize_wandb(args, config_params, config)

    adj_np_train = get_adj_from_object(train_data, device)
    adj_np_valid = get_adj_from_object(valid_data, device)
    adj_np_test = get_adj_from_object(test_data, device)
    fold_size = config.variable_config_dict['dataset_len'] // 6
    adj_np = np.concatenate((adj_np_train[:fold_size * config.variable_config_dict['k_th_fold']], adj_np_valid, adj_np_train[fold_size * (config.variable_config_dict['k_th_fold'] + 1):], adj_np_test), axis=0)

    # TODO: get community data from whole dataset for big runs
    # TODO: mask out the community data for the subsets
    # TODO: get mask from k_th_fold
    get_community_data(
        config=config,
        adj_np=adj_np,
        device=device,
        users_top_percent=args.users_top_percent,
        items_top_percent=args.items_top_percent
    )

    get_biased_connectivity_data(config=config, adj_tens=torch.tensor(adj_np, device=device))

    model = initialize_model(args.model_name, config, train_data)

    best_valid_score, best_valid_result, trainer = train_and_evaluate(
        config=config,
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        use_dropout=args.use_dropout
    )

    # load the folds results from wandb (last 5 runs) including all eval scores and get averages from all folds
    if args.k_th_fold == 5:
        last_5_runs = wandb.Api().runs(path="RecSys_PowerNodeEdgeDropout/RecSys_PowerNodeEdgeDropout", order_by="-created_at", per_page=5)
        all_runs = []



    rng_id = np.random.randint(0, 100000)
    wandb.save(f"{args.model_name}_{args.dataset_name}_ID{rng_id}.h5")
    wandb_run.finish()


if __name__ == "__main__":
    main()


