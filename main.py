import time

import torch
import numpy as np
import os
from collections import defaultdict
import warnings
from config import Config
from evaluation import evaluate_model, print_metric_results, evaluate_model_vectorized, get_community_bias
from models import get_model
from dataset import RecommendationDataset, prepare_adj_tensor
from argparse import ArgumentParser

from plotting import plot_community_bias
from utils_functions import get_community_data, get_biased_connectivity_data, set_seed
import wandb
from wandb_logging import init_wandb, log_metrics_to_wandb
from training import train_model
from scipy import sparse as sp

warnings.filterwarnings('ignore')


def main():
    set_seed(21)  # For reproducibility
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default='MultiVAE')
    parser.add_argument("--dataset_name", type=str, default='ml-100k')
    parser.add_argument("--users_top_percent", type=float, default=0.05)
    parser.add_argument("--items_top_percent", type=float, default=0.05)
    parser.add_argument("--users_dec_perc_suppr", type=float, default=0.3)
    parser.add_argument("--items_dec_perc_suppr", type=float, default=0.0)
    parser.add_argument("--community_suppression", type=float, default=0.5)
    parser.add_argument("--suppress_power_nodes_first", type=str, default='True')
    parser.add_argument("--use_suppression", type=str, default='False')
    # False: 0.4907      0.4827      0.4386      0.3752, cv folds
    # True:  0.4881      0.4795      0.4364      0.3722, cv folds, perc suppression 0.3, strength 0.5
    config = Config()
    config.update_from_args(parser.parse_args())
    config.setup_model_config()
    init_wandb(config, offline=True)

    dataset = RecommendationDataset(name=config.dataset_name, data_path=f'dataset/{config.dataset_name}')
    dataset.prepare_data()

    config.user_degrees, config.item_degrees = dataset.get_node_degrees()
    print("Preparing data with consistent encoding...")
    print(f"Processed data: {dataset.num_users} users, {dataset.num_items} items, {len(dataset.complete_df)} interactions")

    wandb.log({
        'dataset/dataset_name': dataset.name,
        'dataset/num_users': dataset.num_users,
        'dataset/num_items': dataset.num_items,
        'dataset/sparsity': 1 - (len(dataset.complete_df) / (dataset.num_users * dataset.num_items))
    })

    # 5-fold cross validation using temporal splits within training set
    print("\nStarting 5-fold cross validation with complete graph...")
    if config.use_suppression:
        print(f"Community edge suppression ENABLED - suppression strength: {config.community_suppression}")
        print(f"User dropout: {config.users_dec_perc_suppr}, Item dropout: {config.items_dec_perc_suppr}")
    else:
        print("Community edge suppression DISABLED")

    # Pre-calculate community data and biased edges
    adj_np = dataset.complete_df[['user_encoded', 'item_encoded', 'rating']].values
    adj_tens = prepare_adj_tensor(dataset)

    get_community_data(config, adj_np)

    get_biased_connectivity_data(config, adj_tens)

    user_biases, item_biases = get_community_bias(item_communities_each_user_dist=config.item_community_connectivity_matrix_distribution,
                       user_communities_each_item_dist=config.user_community_connectivity_matrix_distribution)
    plot_community_bias(user_biases, item_biases, save_path=f'dataset/{config.dataset_name}/images/', dataset_name=config.dataset_name)

    cv_results = []
    train_time_start = time.time()

    n_folds = 5
    for fold in range(n_folds):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        model = get_model(config=config, dataset=dataset)

        dataset.get_fold_i(i=fold)  # sets train_df and val_df for this fold
        config.train_mask = dataset.train_mask

        model = train_model(dataset=dataset, model=model, config=config, fold_num=fold + 1,)

        val_metrics = evaluate_model_vectorized(
            model=model, dataset=dataset, config=config,
            k_values=config.evaluate_top_k)

        cv_results.append(val_metrics)
        log_metrics_to_wandb(val_metrics, config, stage=f'fold_{fold+1}')
        print_metric_results(val_metrics, f"Fold {fold + 1} Results")

    if cv_results:
        # get average metrics across folds
        cv_summary = {}
        for fold_result in cv_results:
            for k_, metrics in fold_result.items():
                if k_ not in cv_summary:
                    cv_summary[k_] = defaultdict(list)
                for metric_name, value in metrics.items():
                    cv_summary[k_][metric_name].append(value)
        # Calculate averages
        for k_, metrics in cv_summary.items():
            for metric_name, values in metrics.items():
                cv_summary[k_][metric_name] = np.mean(values)

        log_metrics_to_wandb(cv_summary, config, stage='cv_avg')
        print_metric_results(cv_summary, "CROSS-VALIDATION SUMMARY (5-fold average)")

    train_time_end = time.time()
    print(f"\nTraining time: {n_folds} folds, model {config.model_name}, on {config.dataset_name}: {(train_time_end - train_time_start)/60:.0f} minutes")

    model_artifact = wandb.Artifact(
        name=f"model_{config.model_name}_{config.dataset_name}",
        type="model",
        description=f"Trained {config.model_name} model on {config.dataset_name} dataset")

    model_path = "final_model.pth"
    if config.model_name != 'ItemKNN':
        torch.save(model.state_dict(), model_path)
    else:
        # For ItemKNN, we save the model as a sparse matrix
        sp.save_npz(model_path, model.similarity_matrix)
        model_path += '.npz'
    model_artifact.add_file(model_path)

    wandb.log_artifact(model_artifact)

    # print_metric_results(test_metrics, "FINAL TEST SET RESULTS")

    # Clean up temporary files
    if os.path.exists(model_path):
        os.remove(model_path)

    wandb.finish()


if __name__ == "__main__":
    main()

