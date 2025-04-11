from utils import set_seed, power_node_edge_dropout
from precompute import get_community_connectivity_matrix, get_community_labels, get_power_users_items, percent_pointing_inside_com
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
import wandb
from argparse import ArgumentParser
import yaml
import logging
from logging import getLogger
from recbole.utils import init_seed, init_logger
from recbole.utils.case_study import full_sort_topk
from recbole.model.general_recommender import LightGCN, ItemKNN, MultiVAE
import gc
import numpy as np
from recbole.data.interaction import Interaction
from recbole.utils import init_seed
import os
import torch
import copy
from PowerDropoutTrainer import PowerDropoutTrainer
from recbole.trainer import Trainer


def main():
    seed = 42
    set_seed(seed)

    parser = ArgumentParser()
    # # in cmd: python main.py --model_name LightGCN --dataset_name ml-20m --config_file_name ml-20_config.yaml --users_top_percent 0.01 --users_dec_perc_drop 0.70 --community_dropout_strength 0.5 --do_power_nodes_from_community True
    parser.add_argument("--model_name", type=str, default='LightGCN')
    parser.add_argument("--dataset_name", type=str, default='ml-20m')
    # parser.add_argument("--config_file_name", type=str, default=f'ml-20_config.yaml')
    parser.add_argument("--users_top_percent", type=float, default=0.01)
    parser.add_argument("--items_top_percent", type=float, default=0.0)  # isn't item dropout what we want in the end to min bias?
    parser.add_argument("--users_dec_perc_drop", type=float, default=0.70)
    parser.add_argument("--items_dec_perc_drop", type=float, default=0.0)
    parser.add_argument("--community_dropout_strength", type=float, default=0.9)
    parser.add_argument("--do_power_nodes_from_community", type=bool, default=True)
    # TODO: check scientific evidence for parameter existence and values!

    args = parser.parse_args()

    # debugging args dict:
    # args = {'model_name': 'LightGCN', 'dataset_name': 'yoochoose', 'users_top_percent': 0.01, 'users_dec_perc_drop': 0.70, 'community_dropout_strength': 0.5, 'do_power_nodes_from_community': True, 'items_top_percent': 0, 'items_dec_perc_drop': 0}
    model_name = args.model_name
    dataset_name = args.dataset_name
    users_top_percent = args.users_top_percent
    items_top_percent = args.items_top_percent
    users_dec_perc_drop = args.users_dec_perc_drop
    items_dec_perc_drop = args.items_dec_perc_drop
    do_power_nodes_from_community = args.do_power_nodes_from_community
    community_dropout_strength = args.community_dropout_strength

    with open(f'{dataset_name}_config.yaml', 'r') as file:
        config_file = yaml.safe_load(file)
        batch_size = config_file['train_batch_size']
        rating_col_name = config_file['RATING_FIELD']
        topk = config_file['topk']
        epochs = config_file['epochs']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # create_dataset checks device automatically
    torch.set_default_device(device)

    config = Config(model=model_name, dataset=dataset_name, config_file_list=[f'{dataset_name}_config.yaml'], config_dict={'users_dec_perc_drop': users_dec_perc_drop, 'items_dec_perc_drop': items_dec_perc_drop, 'community_dropout_strength': community_dropout_strength})
    init_seed(seed=seed, reproducibility=config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)
    logger.info(config)

    # set torch default device, check if this is already done in recbole.config
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    print(f"cuda: {torch.cuda.is_available()}")
    dataset = create_dataset(config)  # object of shape (n, (user, item, rating))

    # preprocessing dataset
    # thresholding done already in create_dataset() but in case they haven't deleted the edges
    # dataset = booleanify(dataset, threshold=4, rating_col_name=rating_col_name)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # initializing wandb
    wandb.login(key="d234bc98a4761bff39de0e5170df00094ac42269")

    # TODO: change "power items/users" to "item/user hubs"
    wandb_run = wandb.init(
        project="RecSys_PowerNodeEdgeDropout",
        name=f"{model_name}_{dataset_name}_users_top_{users_top_percent}_com_drop_strength_{community_dropout_strength}",
        config={
            "epochs": epochs,
            "dataset": dataset_name,
            "model": model_name,
            "users_top_percent": users_top_percent,
            "items_top_percent": items_top_percent,
            "users_dec_perc_drop": users_dec_perc_drop,
            "items_dec_perc_drop": items_dec_perc_drop,
            "community_dropout_strength": community_dropout_strength,
            "do_power_nodes_from_community": do_power_nodes_from_community,
            "batch_size": batch_size,
            "TopK": topk})

    # make path if not exists
    if not os.path.exists(f'dataset/{dataset_name}'):
        os.makedirs(f'dataset/{dataset_name}')

    ### TODO: instead of preprocessing only the train data, we should preprocess the whole dataset and save it in a file
    # TODO: => use precompute for this. Check if the files already exist, if not, run precompute
    train_data_coo = copy.deepcopy(train_data.dataset).inter_matrix()
    # combine row and col into torch.tensor of shape (n, 2) converting the data to numpy arrays and concatenating them
    indices = torch.tensor((train_data_coo.row, train_data_coo.col), dtype=torch.int32).T
    values = torch.unsqueeze(torch.tensor(train_data_coo.data, dtype=torch.int32), dim=0).T
    adj_np = np.array(torch.cat((indices, values), dim=1), dtype=np.int64)
    del train_data_coo, indices, values

    bipartite_connect = True  # if bipartite community detection, if True: connect items communities to user communities
    if f'user_labels_undir_bip{bipartite_connect}_Leiden.csv' not in os.listdir(f'dataset/{dataset_name}'):
        config.variable_config_dict['user_com_labels'], config.variable_config_dict['item_com_labels'] = get_community_labels(adj_np=adj_np,
                                                          save_path=f'dataset/{dataset_name}',
                                                          get_probs=True)
    else:
        config.variable_config_dict['user_com_labels'] = torch.tensor(np.loadtxt(f'dataset/{dataset_name}/user_labels_undir_bip{bipartite_connect}_Leiden.csv'), dtype=torch.int64)
        config.variable_config_dict['item_com_labels'] = torch.tensor(np.loadtxt(f'dataset/{dataset_name}/item_labels_undir_bip{bipartite_connect}_Leiden.csv'), dtype=torch.int64)

    if f'power_users_ids_com_wise_{do_power_nodes_from_community}_top{users_top_percent}users.csv' not in os.listdir(f'dataset/{dataset_name}') or f'power_items_ids_com_wise_{do_power_nodes_from_community}_top{items_top_percent}items.csv' not in os.listdir(f'dataset/{dataset_name}'):
        config.variable_config_dict['power_users_ids'], config.variable_config_dict['power_items_ids'] = get_power_users_items(
            adj_tens=torch.tensor(adj_np),
            user_com_labels=config.variable_config_dict['user_com_labels'],
            item_com_labels=config.variable_config_dict['item_com_labels'],
            users_top_percent=users_top_percent,
            items_top_percent=items_dec_perc_drop,
            do_power_nodes_from_community=do_power_nodes_from_community,
            save_path=f'dataset/{dataset_name}')
    else:
        config.variable_config_dict['power_users_ids'] = torch.tensor(np.loadtxt(f'dataset/{dataset_name}/power_users_ids_com_wise_{do_power_nodes_from_community}_top{users_top_percent}users.csv'), dtype=torch.int64)

    # change later to a call of precompute
    get_community_connectivity_matrix(adj_np=adj_np, save_path=f'dataset/{dataset_name}')
    percent_pointing_inside_com(adj_np=adj_np,
                                user_com_labels=config.variable_config_dict['user_com_labels'],
                                item_com_labels=config.variable_config_dict['item_com_labels'],
                                save_path=f'dataset/{dataset_name}')


    # tensor with tensor[com_label] = average degree of each community
    config.variable_config_dict['com_avg_dec_degrees'] = torch.zeros(torch.max(config.variable_config_dict['user_com_labels']) + 1)
    adj_tens = torch.tensor(adj_np)
    for com_label in torch.unique(config.variable_config_dict['user_com_labels']):
        nr_nodes_in_com = torch.count_nonzero(config.variable_config_dict['user_com_labels'] == com_label)
        nr_edges_in_com = torch.sum(config.variable_config_dict['user_com_labels'][adj_tens[:, 0]] == com_label)
        decimal_avg_degree_com_label = nr_edges_in_com / nr_nodes_in_com / nr_nodes_in_com
        config.variable_config_dict['com_avg_dec_degrees'][com_label] = decimal_avg_degree_com_label

    if model_name == 'LightGCN':
        model = LightGCN(config, train_data.dataset).to(config['device'])
    elif model_name == 'ItemKNN':
        model = ItemKNN(config, train_data.dataset).to(config['device'])
    elif model_name == 'MultiVAE':
        model = MultiVAE(config, train_data.dataset).to(config['device'])
    else:
        raise ValueError(f"Model {model_name} not supported")
    logger.info(model)

    trainer = PowerDropoutTrainer(config, model)
    # trainer = Trainer(config, model)

    best_valid_score, best_valid_result = trainer.fit(train_data, test_data, saved=True, show_progress=config['show_progress'])

    wandb.log({"best_valid_score": best_valid_score, "best_valid_result": best_valid_result})
    logger.info(f"Best valid score: {best_valid_score}, best valid result: {best_valid_result}")

    # TODO: evaluate custom community bias
    trainer.evaluate(valid_data, show_progress=config['show_progress'])

    ## save model
    rng_id = np.random.randint(0, 100000)
    wandb.save(f"{model_name}_{dataset_name}_ID{rng_id}.h5")
    wandb_run.finish()
    ## del trainer, train_data, valid_data, test_data
    ## gc.collect()  # garbage collection
    #


if __name__ == "__main__":
    main()





