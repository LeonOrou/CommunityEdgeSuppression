import optuna
import wandb
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import LightGCN, ItemKNN, MultiVAE
from recbole.utils import init_seed, init_logger
from PowerDropoutTrainer import PowerDropoutTrainer
from utils_functions import set_seed
from precompute import get_community_connectivity_matrix, get_community_labels, get_power_users_items
import yaml
import torch
import numpy as np
import os
import copy
import logging
from logging import getLogger
import gc

# Constants
SEED = 42
DATASET_NAME = "ml-100k"
N_TRIALS = 30  # Number of hyperparameter combinations to try
USERS_TOP_PERCENT = 0.05
ITEMS_TOP_PERCENT = 0.05


def load_config(dataset_name):
    """Load the configuration file"""
    with open(f'{dataset_name}_config.yaml', 'r') as file:
        config_file = yaml.safe_load(file)
    return config_file


def prepare_data_and_communities(config, device, users_top_percent, items_top_percent,
                                 do_power_nodes_from_community=True):
    """Prepare data and compute communities for the model"""
    init_seed(seed=SEED, reproducibility=config['reproducibility'])
    init_logger(config)
    logger = getLogger()

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    train_data_coo = copy.deepcopy(train_data.dataset).inter_matrix()
    indices = torch.tensor((train_data_coo.row, train_data_coo.col), dtype=torch.int32, device=device).T
    values = torch.unsqueeze(torch.tensor(train_data_coo.data, dtype=torch.int32, device=device), dim=0).T
    adj_np = np.array(torch.cat((indices, values), dim=1).cpu(), dtype=np.int64)
    del train_data_coo, indices, values

    bipartite_connect = True

    # Cache path checking
    if not os.path.exists(f'dataset/{config['dataset']}'):
        os.makedirs(f'dataset/{config['dataset']}')

    # Get or load community labels
    if f'user_labels_undir_bip{bipartite_connect}_Leiden.csv' not in os.listdir(f'dataset/{config['dataset']}'):
        config.variable_config_dict['user_com_labels'], config.variable_config_dict[
            'item_com_labels'] = get_community_labels(
            adj_np=adj_np,
            save_path=f'dataset/{config['dataset']}',
            get_probs=True
        )
    else:
        config.variable_config_dict['user_com_labels'] = torch.tensor(
            np.loadtxt(f'dataset/{config['dataset']}/user_labels_undir_bip{bipartite_connect}_Leiden.csv'),
            dtype=torch.int64, device=device
        )
        config.variable_config_dict['item_com_labels'] = torch.tensor(
            np.loadtxt(f'dataset/{config['dataset']}/item_labels_undir_bip{bipartite_connect}_Leiden.csv'),
            dtype=torch.int64, device=device
        )

    # Get or load power users/items
    if f'power_users_ids_com_wise_{do_power_nodes_from_community}_top{users_top_percent}users.csv' not in os.listdir(
            f'dataset/{config['dataset']}') or \
            f'power_items_ids_com_wise_{do_power_nodes_from_community}_top{items_top_percent}items.csv' not in os.listdir(
        f'dataset/{config['dataset']}'):
        config.variable_config_dict['power_users_ids'], config.variable_config_dict[
            'power_items_ids'] = get_power_users_items(
            adj_tens=torch.tensor(adj_np, device=device),
            user_com_labels=config.variable_config_dict['user_com_labels'],
            item_com_labels=config.variable_config_dict['item_com_labels'],
            users_top_percent=users_top_percent,
            items_top_percent=items_top_percent,
            do_power_nodes_from_community=do_power_nodes_from_community,
            save_path=f'dataset/{config['dataset']}'
        )
    else:
        config.variable_config_dict['power_users_ids'] = torch.tensor(
            np.loadtxt(
                f'dataset/{config['dataset']}/power_users_ids_com_wise_{do_power_nodes_from_community}_top{users_top_percent}users.csv'),
            dtype=torch.int64, device=device
        )
        config.variable_config_dict['power_items_ids'] = torch.tensor(
            np.loadtxt(
                f'dataset/{config['dataset']}/power_items_ids_com_wise_{do_power_nodes_from_community}_top{items_top_percent}items.csv'),
            dtype=torch.int64, device=device
        )

    # Calculate community average degrees
    config.variable_config_dict['com_avg_dec_degrees'] = torch.zeros(
        torch.max(config.variable_config_dict['user_com_labels']) + 1, device=device)
    adj_tens = torch.tensor(adj_np, device=device)
    for com_label in torch.unique(config.variable_config_dict['user_com_labels']):
        nr_nodes_in_com = torch.count_nonzero(config.variable_config_dict['user_com_labels'] == com_label)
        nr_edges_in_com = torch.sum(config.variable_config_dict['user_com_labels'][adj_tens[:, 0]] == com_label)
        # decimal_avg_degree_com_label = nr_edges_in_com / nr_nodes_in_com / nr_nodes_in_com
        # config.variable_config_dict['com_avg_dec_degrees'][com_label] = decimal_avg_degree_com_label

    return train_data, valid_data, test_data


def create_model(model_name, config, train_data):
    """Create a recommendation model based on model name"""
    if model_name == 'LightGCN':
        return LightGCN(config, train_data.dataset).to(config['device'])
    elif model_name == 'ItemKNN':
        return ItemKNN(config, train_data.dataset).to(config['device'])
    elif model_name == 'MultiVAE':
        return MultiVAE(config, train_data.dataset).to(config['device'])
    else:
        raise ValueError(f"Model {model_name} not supported")


def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    # Sample hyperparameters
    torch.cuda.empty_cache()

    model_name = trial.suggest_categorical("model_name", ["LightGCN"])
    users_dec_perc_drop = trial.suggest_categorical("users_dec_perc_drop", [0.0, 0.1])
    items_dec_perc_drop = trial.suggest_categorical("items_dec_perc_drop", [0.0, 0.2])
    community_dropout_strength = trial.suggest_categorical("community_dropout_strength", [0.0, 0.5, 0.9])
    do_power_nodes_from_community = True
    DATASET_NAME = "ml-100k"

    set_seed(SEED)

    # Load base config
    config_file = load_config(DATASET_NAME)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize config
    config = Config(
        model=model_name,
        dataset=DATASET_NAME,
        config_file_list=[f'{DATASET_NAME}_config.yaml'],
        config_dict={
            'users_dec_perc_drop': users_dec_perc_drop,
            'items_dec_perc_drop': items_dec_perc_drop,
            'community_dropout_strength': community_dropout_strength
        }
    )
    config['device'] = device

    # Initialize WandB for this trial
    trial_name = f"{model_name}_{DATASET_NAME}_u{users_dec_perc_drop}_i{items_dec_perc_drop}_c{community_dropout_strength}_trial{trial.number}"
    wandb.init(
        project="RecSys_HPSearch",
        name=trial_name,
        config={
            "epochs": config_file['epochs'],
            "dataset": DATASET_NAME,
            "model": model_name,
            "users_top_percent": USERS_TOP_PERCENT,
            "items_top_percent": ITEMS_TOP_PERCENT,
            "users_dec_perc_drop": users_dec_perc_drop,
            "items_dec_perc_drop": items_dec_perc_drop,
            "community_dropout_strength": community_dropout_strength,
            "do_power_nodes_from_community": do_power_nodes_from_community,
            "batch_size": config_file['train_batch_size'],
            "TopK": config_file['topk'],
            "trial": trial.number
        }
    )

    # Prepare data and communities
    train_data, valid_data, test_data = prepare_data_and_communities(
        config, device, USERS_TOP_PERCENT, ITEMS_TOP_PERCENT, do_power_nodes_from_community
    )

    # Create and train model
    model = create_model(model_name, config, train_data)
    logger = getLogger()
    logger.info(model)

    # Use PowerDropoutTrainer for training (or regular Trainer if needed)
    trainer = PowerDropoutTrainer(config, model)
    # trainer = Trainer(config, model)  # Alternative

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # Log results to WandB
    wandb.log({
        "best_valid_score": best_valid_score,
        **best_valid_result
    })

    # Clean up
    # del model
    # del trainer
    # del model
    del train_data
    del valid_data
    del test_data
    wandb.finish()
    gc.collect()
    torch.cuda.empty_cache()

    # Return metric to optimize (using whichever metric is most important)
    # Assuming we want to maximize NDCG@50
    return best_valid_result.get(f"ndcg@{config_file['topk']}", 0.0)


def run_hyperparameter_search():
    """Run the hyperparameter search"""
    set_seed(SEED)

    # Create an Optuna study that maximizes the objective
    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    # (0.0, 0.2, 0.9)
    # (0.1, 0.0, 0.0)
    # (0.1, 0.0, 0.5)
    specific_combinations = [
        {
            "model_name": "LightGCN",
            "users_dec_perc_drop": 0.1,
            "items_dec_perc_drop": 0.0,
            "community_dropout_strength": 0.0
        },
        {
            "model_name": "LightGCN",
            "users_dec_perc_drop": 0.0,
            "items_dec_perc_drop": 0.2,
            "community_dropout_strength": 0.9
        },
        {
            "model_name": "LightGCN",
            "users_dec_perc_drop": 0.1,
            "items_dec_perc_drop": 0.0,
            "community_dropout_strength": 0.5
        }
    ]
    for params in specific_combinations:
        study.enqueue_trial(params)
    study.optimize(objective, n_trials=N_TRIALS)

    optuna.visualization.plot_param_importances(study).write_html("param_importances.html")
    optuna.visualization.plot_optimization_history(study).write_html("optimization_history.html")

    # Print results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Log best hyperparameters to a file
    with open(f"best_hyperparams_{DATASET_NAME}.yaml", "w") as f:
        yaml.dump(trial.params, f)


if __name__ == "__main__":
    run_hyperparameter_search()

