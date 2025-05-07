from utils_functions import set_seed, plot_community_confidence, plot_community_connectivity_distribution, plot_degree_distributions
from precompute import get_community_connectivity_matrix, get_community_labels, get_power_users_items
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


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser()
    # in cmd: python main.py --model_name LightGCN --dataset_name ml-20m --config_file_name ml-20_config.yaml --users_top_percent 0.01 --users_dec_perc_drop 0.70 --community_dropout_strength 0.5 --do_power_nodes_from_community True
    parser.add_argument("--model_name", type=str, default='LightGCN')
    parser.add_argument("--dataset_name", type=str, default='ml-100k')
    parser.add_argument("--users_top_percent", type=float, default=0.01)
    parser.add_argument("--items_top_percent", type=float, default=0.05)
    parser.add_argument("--users_dec_perc_drop", type=float, default=0.0)
    parser.add_argument("--items_dec_perc_drop", type=float, default=0.1)
    parser.add_argument("--community_dropout_strength", type=float, default=0.6)
    parser.add_argument("--do_power_nodes_from_community", type=bool, default=True)
    # parser.add_argument("--do_power_nodes_from_community", action="store_true")
    # TODO: check scientific evidence for parameter existence and values!
    return parser.parse_args()


def load_config_from_yaml(dataset_name):
    """Load configuration from YAML file."""
    with open(f'{dataset_name}_config.yaml', 'r') as file:
        config_file = yaml.safe_load(file)
        return {
            'batch_size': config_file['train_batch_size'],
            'rating_col_name': config_file['RATING_FIELD'],
            'topk': config_file['topk'],
            'epochs': config_file['epochs']
        }


def setup_device(try_gpu=True):
    """Setup device (CPU/GPU)."""
    if try_gpu:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device('cpu')


def setup_config(args, device, seed):
    """Setup RecBole configuration."""
    config = Config(
        model=args.model_name,
        dataset=args.dataset_name,
        config_file_list=[f'{args.dataset_name}_config.yaml'],
        config_dict={
            'users_dec_perc_drop': args.users_dec_perc_drop,
            'items_dec_perc_drop': args.items_dec_perc_drop,
            'community_dropout_strength': args.community_dropout_strength
        }
    )
    config['device'] = device

    # Initialize seed and logging
    init_seed(seed=seed, reproducibility=config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)
    logger.info(config)
    
    return config, logger


def prepare_dataset(config):
    """Create and prepare dataset."""
    dataset = create_dataset(config)
    logger = getLogger()
    logger.info(dataset)
    return data_preparation(config, dataset)  # outputs train_data, valid_data, test_data


def initialize_wandb(args, config_params):
    """Initialize Weights & Biases for experiment tracking."""
    wandb.login(key="d234bc98a4761bff39de0e5170df00094ac42269")
    return wandb.init(
        project="RecSys_PowerNodeEdgeDropout",
        name=f"{args.model_name}_{args.dataset_name}_users_top_{args.users_top_percent}_com_drop_strength_{args.community_dropout_strength}",
        config={
            "epochs": config_params['epochs'],
            "dataset": args.dataset_name,
            "model": args.model_name,
            "users_top_percent": args.users_top_percent,
            "items_top_percent": args.items_top_percent,
            "users_dec_perc_drop": args.users_dec_perc_drop,
            "items_dec_perc_drop": args.items_dec_perc_drop,
            "community_dropout_strength": args.community_dropout_strength,
            "do_power_nodes_from_community": args.do_power_nodes_from_community,
            "batch_size": config_params['batch_size'],
            "TopK": config_params['topk']
        }
    )


def preprocess_train_data(train_data, device):
    """Preprocess training data to adjacency matrix format."""
    train_data_coo = copy.deepcopy(train_data.dataset).inter_matrix()
    indices = torch.tensor((train_data_coo.row, train_data_coo.col), dtype=torch.int32, device=device).T
    values = torch.unsqueeze(torch.tensor(train_data_coo.data, dtype=torch.int32, device=device), dim=0).T
    adj_np = np.array(torch.cat((indices, values), dim=1).cpu(), dtype=np.int64)
    return adj_np


def get_or_load_community_data(config, dataset_name, adj_np, device, do_power_nodes_from_community, users_top_percent, items_top_percent):
    """Get or load community labels and power nodes."""
    # Create directory if it doesn't exist
    if not os.path.exists(f'dataset/{dataset_name}'):
        os.makedirs(f'dataset/{dataset_name}')
    
    bipartite_connect = True  # if bipartite community detection, if True: connect items communities to user communities
    
    # Get or load community labels
    if f'user_labels_Leiden.csv' not in os.listdir(f'dataset/{dataset_name}'):
        config.variable_config_dict['user_com_labels'], config.variable_config_dict['item_com_labels'] = get_community_labels(
            adj_np=adj_np,
            save_path=f'dataset/{dataset_name}',
            get_probs=True
        )
    else:
        config.variable_config_dict['user_com_labels'] = torch.tensor(
            np.loadtxt(f'dataset/{dataset_name}/user_labels_Leiden.csv', dtype=np.int64),
            dtype=torch.int64, 
            device=device
        )
        config.variable_config_dict['item_com_labels'] = torch.tensor(
            np.loadtxt(f'dataset/{dataset_name}/item_labels_Leiden.csv', dtype=np.int64),
            dtype=torch.int64, 
            device=device
        )
    
    # Get or load power nodes
    power_users_file = f'power_users_ids_com_wise_{do_power_nodes_from_community}_top{users_top_percent}users.csv'
    power_items_file = f'power_items_ids_com_wise_{do_power_nodes_from_community}_top{items_top_percent}items.csv'
    
    if power_users_file not in os.listdir(f'dataset/{dataset_name}') or power_items_file not in os.listdir(f'dataset/{dataset_name}'):
        config.variable_config_dict['power_users_ids'], config.variable_config_dict['power_items_ids'] = get_power_users_items(
            adj_tens=torch.tensor(adj_np, device=device),
            user_com_labels=config.variable_config_dict['user_com_labels'],
            item_com_labels=config.variable_config_dict['item_com_labels'],
            users_top_percent=users_top_percent,
            items_top_percent=items_top_percent,
            do_power_nodes_from_community=do_power_nodes_from_community,
            save_path=f'dataset/{dataset_name}'
        )
    else:
        config.variable_config_dict['power_users_ids'] = torch.tensor(
            np.loadtxt(f'dataset/{dataset_name}/{power_users_file}'), 
            dtype=torch.int64, 
            device=device
        )
        if os.path.exists(f'dataset/{dataset_name}/{power_items_file}'):
            config.variable_config_dict['power_items_ids'] = torch.tensor(
                np.loadtxt(f'dataset/{dataset_name}/{power_items_file}'), 
                dtype=torch.int64, 
                device=device
            )


def calculate_community_metrics(config, adj_np, device):
    """Calculate community connectivity matrix and average degrees."""
    adj_tens = torch.tensor(adj_np, device=device)
    
    # Get community connectivity matrix
    community_connectivity_matrix = get_community_connectivity_matrix(
        adj_tens=adj_tens,
        user_com_labels=config.variable_config_dict['user_com_labels'],
        item_com_labels=config.variable_config_dict['item_com_labels']
    )
    
    # Calculate average degree for each community
    config.variable_config_dict['com_avg_dec_degrees'] = torch.zeros(
        torch.max(config.variable_config_dict['user_com_labels']) + 1, 
        device=device
    )
    
    for com_label in torch.unique(config.variable_config_dict['user_com_labels']):
        nr_nodes_in_com = torch.count_nonzero(config.variable_config_dict['user_com_labels'] == com_label)
        nr_edges_in_com = torch.sum(config.variable_config_dict['user_com_labels'][adj_tens[:, 0]] == com_label)
        # decimal_avg_degree_com_label = nr_edges_in_com / nr_nodes_in_com / nr_nodes_in_com
        # config.variable_config_dict['com_avg_dec_degrees'][com_label] = decimal_avg_degree_com_label
    
    return community_connectivity_matrix


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


def train_and_evaluate(config, model, train_data, valid_data, test_data, use_power_dropout=True):
    """Train and evaluate the model."""
    if use_power_dropout:
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


def main():
    # Set seed for reproducibility
    seed = 42
    set_seed(seed)

    args = parse_arguments()

    config_params = load_config_from_yaml(args.dataset_name)

    device = setup_device(try_gpu=True)

    config, logger = setup_config(args, device, seed)

    train_data, valid_data, test_data = prepare_dataset(config)

    wandb_run = initialize_wandb(args, config_params)

    adj_np = preprocess_train_data(train_data, device)

    get_or_load_community_data(
        config=config,
        dataset_name=args.dataset_name,
        adj_np=adj_np,
        device=device,
        do_power_nodes_from_community=args.do_power_nodes_from_community,
        users_top_percent=args.users_top_percent,
        items_top_percent=args.items_top_percent
    )

    community_connectivity_matrix = calculate_community_metrics(config, adj_np, device)
    
    # Optional: Uncomment for plots
    # plot_degree_distributions(adj_tens=torch.tensor(adj_np, device=device), num_bins=100, save_path=f'images/', dataset_name=args.dataset_name)
    # plot_community_connectivity_distribution(connectivity_matrix=community_connectivity_matrix, top_n_communities=20, save_path=f'images/', dataset_name=args.dataset_name)
    # plot_community_confidence(user_probs_path=f'', save_path=f'images/', dataset_name=args.dataset_name, top_n_communities=10)

    model = initialize_model(args.model_name, config, train_data)

    best_valid_score, best_valid_result, trainer = train_and_evaluate(
        config=config,
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        use_power_dropout=True  # set false to get default trainer object
    )
    
    # Save model
    rng_id = np.random.randint(0, 100000)
    wandb.save(f"{args.model_name}_{args.dataset_name}_ID{rng_id}.h5")

    wandb_run.finish()


if __name__ == "__main__":
    main()
