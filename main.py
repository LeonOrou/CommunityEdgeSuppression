import torch
import numpy as np
import os
from collections import defaultdict
import warnings
from config import Config
from evaluation import evaluate_model, evaluate_current_model_ndcg, print_metric_results
from models import calculate_bpr_loss, get_model
import torch.optim as optim
from dataset import RecommendationDataset, sample_negative_items, prepare_adj_tensor, prepare_training_data
from argparse import ArgumentParser
from utils_functions import (
    community_edge_suppression, get_community_data, get_biased_connectivity_data, set_seed, )
import wandb
from wandb_logging import init_wandb, log_fold_metrics_to_wandb, log_test_metrics_to_wandb, log_cv_summary_to_wandb
import time
from scipy.sparse import csr_matrix
from training import train_model
from scipy import sparse as sp

warnings.filterwarnings('ignore')


def main():
    args = parse_arguments()
    config = Config()
    config.update_from_args(args)
    config.setup_model_config()
    init_wandb(config)

    dataset = RecommendationDataset(name=config.dataset_name, data_path=f'dataset/{config.dataset_name}')
    dataset.prepare_data()

    config.user_degrees, config.item_degrees = dataset.get_node_degrees()
    print("Preparing data with consistent encoding...")
    print(
        f"Processed data: {dataset.num_users} users, {dataset.num_items} items, {len(dataset.complete_df)} interactions")

    print(f"Train set: {len(dataset.train_val_df)} interactions")
    print(f"Test set: {len(dataset.test_df)} interactions")

    # Log dataset statistics
    wandb.log({
        'dataset/dataset_name': dataset.name,
        'dataset/num_users': dataset.num_users,
        'dataset/num_items': dataset.num_items,
        'dataset/total_interactions': len(dataset.complete_df),
        'dataset/train_interactions': len(dataset.train_val_df),
        'dataset/test_interactions': len(dataset.test_df),
        'dataset/sparsity': 1 - (len(dataset.complete_df) / (dataset.num_users * dataset.num_items))
    })

    # 5-fold cross validation using temporal splits within training set
    print("\nStarting 5-fold cross validation with complete graph...")
    if config.use_dropout:
        print(f"Community edge suppression ENABLED - suppression strength: {config.community_suppression}")
        print(f"User dropout: {config.users_dec_perc_drop}, Item dropout: {config.items_dec_perc_drop}")
    else:
        print("Community edge suppression DISABLED")

    # Pre-calculate community data and biased edges
    adj_np = dataset.complete_df[['user_encoded', 'item_encoded', 'rating']].values
    adj_tens = prepare_adj_tensor(dataset)

    get_community_data(config, adj_np)

    get_biased_connectivity_data(config, adj_tens)

    cv_results = []

    n_folds = 5
    for fold in range(n_folds):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        model = get_model(config=config, dataset=dataset)

        dataset.get_fold_i(i=fold)  # sets train_df and val_df for this fold

        model = train_model(dataset=dataset, model=model, config=config, stage='cv', fold_num=fold + 1,)

        if len(dataset.val_df) > 0:
            val_metrics = evaluate_model(
                model=model, dataset=dataset, config=config,
                k_values=config.evaluate_top_k, stage='cv')


            cv_results.append(val_metrics)
            log_fold_metrics_to_wandb(fold + 1, val_metrics, config)
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

        log_cv_summary_to_wandb(cv_summary, config)
        print_metric_results(cv_summary, "CROSS-VALIDATION SUMMARY (5-fold average)")
    else:
        cv_summary = {}

    # Train final model on all data and evaluate on test set
    print("\n" + "=" * 85)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("=" * 85)

    print("Training final model on full dataset...")

    model = get_model(config=config, dataset=dataset)

    model = train_model(dataset=dataset, model=model, config=config, stage='full_train', fold_num=None,)

    test_metrics = evaluate_model(
        model=model, dataset=dataset, config=config,
        k_values=config.evaluate_top_k, stage='full_train')

    log_test_metrics_to_wandb(test_metrics, config)

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

    print_metric_results(test_metrics, "FINAL TEST SET RESULTS")

    # Clean up temporary files
    if os.path.exists(model_path):
        os.remove(model_path)

    wandb.finish()

    return cv_results, cv_summary, test_metrics, model


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default='ItemKNN',)
    parser.add_argument("--dataset_name", type=str, default='lfm')
    parser.add_argument("--users_top_percent", type=float, default=0.05)
    parser.add_argument("--items_top_percent", type=float, default=0.05)
    parser.add_argument("--users_dec_perc_drop", type=float, default=0.05)
    parser.add_argument("--items_dec_perc_drop", type=float, default=0.05)
    parser.add_argument("--community_suppression", type=float, default=0.6)
    parser.add_argument("--drop_only_power_nodes", type=bool, default=True)
    parser.add_argument("--use_dropout", type=bool, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    set_seed(42)  # For reproducibility
    cv_results, cv_summary, test_metrics, model = main()

