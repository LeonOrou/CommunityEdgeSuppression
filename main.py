import torch
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import warnings
from config import Config
from evaluation import evaluate_model, calculate_ndcg, evaluate_current_model_ndcg
from models import calculate_bpr_loss, multivae_loss, get_model
import torch.optim as optim
from dataset import RecommendationDataset, sample_negative_items, prepare_adj_tensor, prepare_training_data_gpu
from argparse import ArgumentParser
from utils_functions import (
    community_edge_suppression, get_community_data, get_biased_connectivity_data, set_seed,)
import wandb
from wanb_logging import init_wandb, log_fold_metrics_to_wandb, log_test_metrics_to_wandb, log_cv_summary_to_wandb

warnings.filterwarnings('ignore')


def train_model(dataset, model, config, stage='cv', fold_num=None, verbose=True):
    """
    Train LightGCN using the COMPLETE graph structure (train+val+test)
    but only compute loss on training interactions
    stage: 'full_train' for full training evaluation, 'cv' for leave-one-out evaluation
    fold_num: Current fold number for logging (None for final training)
    """
    print(f"Complete graph: {len(dataset.complete_df)} interactions")
    print(f"Training interactions: {len(dataset.train_df)} (used for loss)")
    print(f"Val interactions: {len(dataset.val_df)} (used for evaluation)")
    print(f"Test interactions: {len(dataset.test_df)} (used for evaluation)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # max ndcg
        factor=0.7,  # less aggressive reduction
        patience=20,  # more patience
        min_lr=1e-6,
        verbose=True  # print when LR is reduced
    )

    # Move data to device
    dataset.complete_edge_index = dataset.complete_edge_index.to(device)
    dataset.complete_edge_weight = dataset.complete_edge_weight.to(device)

    # Create user positive items mapping for better negative sampling (from ALL interactions)
    user_positive_items = defaultdict(set)
    for user_id, group in dataset.complete_df.groupby('user_encoded')['item_encoded']:
        user_positive_items[user_id] = set(group.values)

    # Training parameters
    batch_size = config.batch_size
    best_val_ndcg = 0
    patience = config.patience if hasattr(config, 'patience') else 50  # increase default patience
    patience_counter = 0

    # Track validation history for better early stopping
    val_history = []
    best_epoch = 0

    if stage == 'full_train':
        dataset.train_df = dataset.train_val_df  # Use all training+validation data for training
        dataset.val_df = dataset.test_df  # Use test set as validation for final evaluation

    # Pre-convert training data to GPU
    # TODO: make big batches with 2 million interactions maximum and have only this one big batch on GPU
    # TODO get smaller batches with batch_size = config.batch_size from this big batch
    all_train_users, all_train_items, train_indices = prepare_training_data_gpu(dataset.train_df, device)
    num_train = len(train_indices)

    adj_tens = prepare_adj_tensor(dataset)

    model.train()

    for epoch in range(config.epochs):
        # Apply community edge suppression if enabled
        if config.use_dropout:
            # Apply community edge suppression to get modified edge weights
            edge_weights_modified = community_edge_suppression(adj_tens, config)

            if config.model_name == 'LightGCN':
                # undirected graph needed for GNN message propagation
                edge_weights_modified = torch.concatenate((edge_weights_modified, edge_weights_modified))
            # Use the modified weights for this epoch
            current_edge_weight = edge_weights_modified.to(device)
        else:
            # Use original edge weights
            current_edge_weight = dataset.complete_edge_weight

        dataset.current_edge_weight = current_edge_weight

        total_loss = 0
        num_batches = 0

        # Shuffle indices on GPU
        perm = torch.randperm(num_train, device=device)
        train_indices_shuffled = train_indices[perm]

        for i in range(0, num_train, batch_size):
            batch_indices = train_indices_shuffled[i:i + batch_size]

            if len(batch_indices) == 0:
                continue

            batch_users = all_train_users[batch_indices]
            batch_pos_items = all_train_items[batch_indices]

            batch_neg_items = sample_negative_items(
                batch_users,
                batch_pos_items,
                dataset.num_items,
                user_positive_items,
                device
            )

            # Forward pass
            user_emb, item_emb = model(dataset.complete_edge_index, dataset.current_edge_weight)

            batch_user_emb = user_emb[batch_users.long()]
            batch_pos_item_emb = item_emb[batch_pos_items.long()]
            batch_neg_item_emb = item_emb[batch_neg_items.long()]

            bpr_loss = calculate_bpr_loss(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)
            # Add L2 regularization
            l2_reg = config.reg * (
                    batch_user_emb.norm(2).pow(2) +
                    batch_pos_item_emb.norm(2).pow(2) +
                    batch_neg_item_emb.norm(2).pow(2)
            ) / batch_size

            loss = bpr_loss + l2_reg

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            # Use original edge weights for evaluation (no suppression)
            val_ndcg = evaluate_current_model_ndcg(model, dataset, k=10)
            val_history.append(val_ndcg)

            # scheduler.step(val_ndcg)
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / max(num_batches, 1)

            # WandB logging for training metrics
            log_dict = {
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_ndcg@10': val_ndcg,
                'learning_rate': current_lr,
                'patience_counter': patience_counter,
                'suppression_enabled': config.use_dropout
            }

            # Add fold-specific prefix if in cross-validation
            if fold_num is not None:
                log_dict = {f'fold_{fold_num}/{k}': v for k, v in log_dict.items()}
            else:
                log_dict = {f'final_training/{k}': v for k, v in log_dict.items()}

            wandb.log(log_dict)

            if verbose:
                suppression_status = "ON" if config.use_dropout else "OFF"
                print(f'  Epoch {epoch:3d}/{config.epochs}, Loss: {avg_loss:.4f}, '
                      f'Val NDCG@10: {val_ndcg:.4f} (Suppression: {suppression_status})')

            # Improved early stopping with relative improvement check
            if val_ndcg > best_val_ndcg:
                improvement = (val_ndcg - best_val_ndcg) / best_val_ndcg if best_val_ndcg > 0 else 1.0

                # Only update best if improvement is significant (> 0.1%)
                if improvement > 0.001 or best_val_ndcg == 0:
                    best_val_ndcg = val_ndcg
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    best_epoch = epoch
                else:
                    patience_counter += 1
            else:
                patience_counter += 1

            # Additional check: stop if learning has plateaued
            if len(val_history) >= 5:
                recent_std = np.std(val_history[-5:])
                if recent_std < 0.0001 and patience_counter >= patience // 2:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch} - validation plateaued")
                    break

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} - no improvement for {patience} checks")
                    print(f"  Best epoch was {best_epoch} with NDCG@10: {best_val_ndcg:.4f}")
                break

    # Load best model
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)

    # Log final training results
    final_log_dict = {
        'best_epoch': best_epoch,
        'best_val_ndcg@10': best_val_ndcg,
        'total_epochs': epoch + 1,
        'early_stopped': patience_counter >= patience
    }

    if fold_num is not None:
        final_log_dict = {f'fold_{fold_num}/final_{k}': v for k, v in final_log_dict.items()}
    else:
        final_log_dict = {f'final_training/final_{k}': v for k, v in final_log_dict.items()}

    wandb.log(final_log_dict)

    return model, dataset.complete_edge_index, dataset.complete_edge_weight


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

    # Get community labels and power nodes
    get_community_data(config, adj_np)

    # Get biased connectivity data
    get_biased_connectivity_data(config, adj_tens)

    cv_results = []

    n_folds = 5
    for fold in range(n_folds):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        model = get_model(config=config, dataset=dataset)

        dataset.get_fold_i(i=fold)  # sets train_df and val_df for this fold

        model, complete_edge_index, complete_edge_weight = train_model(
            dataset=dataset,
            model=model,
            config=config,
            stage='cv',
            fold_num=fold + 1,  # Pass fold number for logging
        )

        if len(dataset.val_df) > 0:
            val_metrics = evaluate_model(
                model=model, dataset=dataset,
                k_values=config.evaluate_top_k, stage='cv'
            )

            fold_results = {
                'fold': fold + 1,
            }

            # Store metrics for each k - UPDATED TO INCLUDE NEW METRICS
            for k in config.evaluate_top_k:
                fold_results[f'val_ndcg@{k}'] = val_metrics[k]['ndcg']
                fold_results[f'val_recall@{k}'] = val_metrics[k]['recall']
                fold_results[f'val_precision@{k}'] = val_metrics[k]['precision']
                fold_results[f'val_mrr@{k}'] = val_metrics[k]['mrr']
                fold_results[f'val_hit_rate@{k}'] = val_metrics[k]['hit_rate']
                fold_results[f'val_item_coverage@{k}'] = val_metrics[k]['item_coverage']
                fold_results[f'val_gini_index@{k}'] = val_metrics[k]['gini_index']
                fold_results[f'val_simpson_index@{k}'] = val_metrics[k]['simpson_index']

                # NEW METRICS
                fold_results[f'val_simpson_index_genre@{k}'] = val_metrics[k]['simpson_index_genre']
                fold_results[f'val_intra_list_diversity@{k}'] = val_metrics[k]['intra_list_diversity']
                fold_results[f'val_popularity_lift@{k}'] = val_metrics[k]['popularity_lift']
                fold_results[f'val_normalized_genre_entropy@{k}'] = val_metrics[k]['normalized_genre_entropy']
                fold_results[f'val_unique_genres_count@{k}'] = val_metrics[k]['unique_genres_count']
                fold_results[f'val_popularity_calibration@{k}'] = val_metrics[k]['popularity_calibration']

            cv_results.append(fold_results)

            # Log fold results to wandb
            log_fold_metrics_to_wandb(fold + 1, fold_results, config)

            print(f"\nFold {fold + 1} Results:")
            print(f"{'Metric':<20} {'k=10':>10} {'k=20':>10} {'k=50':>10} {'k=100':>10}")
            print("-" * 65)

            # Show key accuracy metrics during training
            for metric, key in [('NDCG', 'ndcg'), ('Recall', 'recall'), ('Hit Rate', 'hit_rate')]:
                row = f"{metric:<20}"
                for k in config.evaluate_top_k:
                    row += f"{val_metrics[k][key]:>10.4f}"
                print(row)

            print("-" * 65)
            # Show key diversity metrics during training
            for metric, key in [('Simpson (Genre)', 'simpson_index_genre'),
                                ('Intra-list Div', 'intra_list_diversity'),
                                ('Genre Entropy', 'normalized_genre_entropy')]:
                row = f"{metric:<20}"
                for k in config.evaluate_top_k:
                    row += f"{val_metrics[k][key]:>10.4f}"
                print(row)

    # Calculate average CV results - UPDATED TO INCLUDE NEW METRICS
    if cv_results:
        cv_df = pd.DataFrame(cv_results)
        cv_summary = {}

        # Calculate mean for each metric at each k
        for k in config.evaluate_top_k:
            cv_summary[k] = {
                'NDCG': cv_df[f'val_ndcg@{k}'].mean(),
                'Recall': cv_df[f'val_recall@{k}'].mean(),
                'Precision': cv_df[f'val_precision@{k}'].mean(),
                'MRR': cv_df[f'val_mrr@{k}'].mean(),
                'Hit Rate': cv_df[f'val_hit_rate@{k}'].mean(),
                'Item Coverage': cv_df[f'val_item_coverage@{k}'].mean(),
                'Gini Index': cv_df[f'val_gini_index@{k}'].mean(),
                'Simpson Index': cv_df[f'val_simpson_index@{k}'].mean(),

                # NEW METRICS
                'Simpson (Genre)': cv_df[f'val_simpson_index_genre@{k}'].mean(),
                'Intra-list Diversity': cv_df[f'val_intra_list_diversity@{k}'].mean(),
                'Popularity Lift': cv_df[f'val_popularity_lift@{k}'].mean(),
                'Genre Entropy': cv_df[f'val_normalized_genre_entropy@{k}'].mean(),
                'Unique Genres': cv_df[f'val_unique_genres_count@{k}'].mean(),
                'Pop. Calibration': cv_df[f'val_popularity_calibration@{k}'].mean(),
            }

        # Log CV summary to wandb
        log_cv_summary_to_wandb(cv_summary, config)

        print("\n" + "=" * 85)
        print("CROSS-VALIDATION SUMMARY (5-fold average)")
        print("=" * 85)

        # Create table with metrics as rows and k values as columns
        print(f"\n{'Metric':<20} {'k=10':>12} {'k=20':>12} {'k=50':>12} {'k=100':>12}")
        print("-" * 80)

        # Accuracy metrics
        accuracy_metrics = ['NDCG', 'Recall', 'Precision', 'MRR', 'Hit Rate']
        for metric in accuracy_metrics:
            row = f"{metric:<20}"
            for k in config.evaluate_top_k:
                value = cv_summary[k][metric]
                row += f"{value:>12.4f}"
            print(row)

        print("-" * 80)

        # Coverage and distribution metrics
        coverage_metrics = ['Item Coverage', 'Gini Index', 'Simpson Index']
        for metric in coverage_metrics:
            row = f"{metric:<20}"
            for k in config.evaluate_top_k:
                value = cv_summary[k][metric]
                row += f"{value:>12.4f}"
            print(row)

        print("-" * 80)

        # New diversity metrics
        diversity_metrics = ['Simpson (Genre)', 'Intra-list Diversity', 'Genre Entropy', 'Unique Genres']
        for metric in diversity_metrics:
            row = f"{metric:<20}"
            for k in config.evaluate_top_k:
                value = cv_summary[k][metric]
                row += f"{value:>12.4f}"
            print(row)

        print("-" * 80)

        # Popularity metrics
        popularity_metrics = ['Popularity Lift', 'Pop. Calibration']
        for metric in popularity_metrics:
            row = f"{metric:<20}"
            for k in config.evaluate_top_k:
                value = cv_summary[k][metric]
                row += f"{value:>12.4f}"
            print(row)

    else:
        cv_summary = {}

    # Train final model on all data and evaluate on test set
    print("\n" + "=" * 85)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("=" * 85)

    print("Training final model with complete graph structure...")

    model = get_model(config=config, dataset=dataset)
    # Final training: use ALL data in graph, but only train+val for loss
    final_model, final_edge_index, final_edge_weight = train_model(
        model=model, dataset=dataset, config=config, stage='full_train',
        fold_num=None,  # No fold number for final training
    )

    # UPDATED TO INCLUDE DATASET_NAME
    test_metrics = evaluate_model(
        model=final_model, dataset=dataset,
        k_values=config.evaluate_top_k, stage='full_train'
    )

    # Log test metrics to wandb
    log_test_metrics_to_wandb(test_metrics, config)

    # Save model artifact to wandb
    model_artifact = wandb.Artifact(
        name=f"model_{config.model_name}_{config.dataset_name}",
        type="model",
        description=f"Trained {config.model_name} model on {config.dataset_name} dataset"
    )

    # Save model state dict
    model_path = "final_model.pth"
    torch.save(final_model.state_dict(), model_path)
    model_artifact.add_file(model_path)

    # Save model architecture info
    model_info = {
        'model_class': config.model_name,
        'num_users': dataset.num_users,
        'num_items': dataset.num_items,
        'embedding_dim': config.embedding_dim,
        'num_layers': config.n_layers if config.model_name == 'LightGCN' else None,
        'total_parameters': sum(p.numel() for p in final_model.parameters()),
        'trainable_parameters': sum(p.numel() for p in final_model.parameters() if p.requires_grad)}

    wandb.log({
        'model/total_parameters': model_info['total_parameters'],
        'model/trainable_parameters': model_info['trainable_parameters']})

    wandb.log_artifact(model_artifact)

    print(f"\nFINAL TEST SET RESULTS")
    print("=" * 85)

    # Create table with metrics as rows and k values as columns
    print(f"\n{'Metric':<20} {'k=10':>12} {'k=20':>12} {'k=50':>12} {'k=100':>12}")
    print("-" * 80)

    # UPDATED METRICS DICTIONARY TO INCLUDE NEW METRICS
    accuracy_metrics_dict = {
        'NDCG': 'ndcg',
        'Recall': 'recall',
        'Precision': 'precision',
        'MRR': 'mrr',
        'Hit Rate': 'hit_rate'
    }

    coverage_metrics_dict = {
        'Item Coverage': 'item_coverage',
        'Gini Index': 'gini_index',
        'Simpson Index': 'simpson_index'
    }

    diversity_metrics_dict = {
        'Simpson (Genre)': 'simpson_index_genre',
        'Intra-list Diversity': 'intra_list_diversity',
        'Genre Entropy': 'normalized_genre_entropy',
        'Unique Genres': 'unique_genres_count'
    }

    popularity_metrics_dict = {
        'Popularity Lift': 'popularity_lift',
        'Pop. Calibration': 'popularity_calibration'
    }

    # Print accuracy metrics
    for display_name, metric_key in accuracy_metrics_dict.items():
        row = f"{display_name:<20}"
        for k in config.evaluate_top_k:
            value = test_metrics[k][metric_key]
            row += f"{value:>12.4f}"
        print(row)

    print("-" * 80)

    # Print coverage metrics
    for display_name, metric_key in coverage_metrics_dict.items():
        row = f"{display_name:<20}"
        for k in config.evaluate_top_k:
            value = test_metrics[k][metric_key]
            row += f"{value:>12.4f}"
        print(row)

    print("-" * 80)

    # Print diversity metrics
    for display_name, metric_key in diversity_metrics_dict.items():
        row = f"{display_name:<20}"
        for k in config.evaluate_top_k:
            value = test_metrics[k][metric_key]
            row += f"{value:>12.4f}"
        print(row)

    print("-" * 80)

    # Print popularity metrics
    for display_name, metric_key in popularity_metrics_dict.items():
        row = f"{display_name:<20}"
        for k in config.evaluate_top_k:
            value = test_metrics[k][metric_key]
            row += f"{value:>12.4f}"
        print(row)

    # Log final summary statistics
    wandb.log({
        'experiment/total_folds': n_folds,
        'experiment/best_cv_ndcg@10': max([cv_results[i]['val_ndcg@10'] for i in range(len(cv_results))]),
        'experiment/final_test_ndcg@10': test_metrics[10]['ndcg'],
        'experiment/cv_std_ndcg@10': cv_df['val_ndcg@10'].std() if cv_results else 0,
        'experiment/completed': True})

    # Clean up temporary files
    if os.path.exists(model_path):
        os.remove(model_path)

    wandb.finish()

    return cv_results, cv_summary, test_metrics, final_model


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default='LightGCN')
    parser.add_argument("--dataset_name", type=str, default='ml-100k')
    parser.add_argument("--users_top_percent", type=float, default=0.05)
    parser.add_argument("--items_top_percent", type=float, default=0.05)
    parser.add_argument("--users_dec_perc_drop", type=float, default=0.05)
    parser.add_argument("--items_dec_perc_drop", type=float, default=0.05)
    parser.add_argument("--community_suppression", type=float, default=0.4)
    parser.add_argument("--drop_only_power_nodes", type=bool, default=True)
    parser.add_argument("--use_dropout", type=bool, default=True)

    # model hyperparameters
    # parser.add_argument("--embedding_dim", type=int, default=128)
    # parser.add_argument("--n_layers", type=int, default=3)
    # parser.add_argument("--item_knn_topk", type=int, default=250)
    # parser.add_argument("--shrink", type=int, default=10)
    # parser.add_argument("--batch_size", type=int, default=512)
    # parser.add_argument("--epochs", type=int, default=100)
    # parser.add_argument("--learning_rate", type=float, default=0.005)
    # parser.add_argument("--hidden_dimension", type=int, default=256)
    # parser.add_argument("--latent_dimension", type=int, default=64)
    # parser.add_argument("--anneal_cap", type=float, default=0.2)

    return parser.parse_args()


if __name__ == "__main__":
    set_seed(42)  # For reproducibility
    cv_results, cv_summary, test_metrics, final_model = main()

    print("\n" + "=" * 85)
    print("DETAILED FOLD RESULTS")
    print("=" * 85)

    for result in cv_results:
        print(f"\nFold {result['fold']}:")
        print(f"{'Metric':<20} {'k=10':>12} {'k=20':>12} {'k=50':>12} {'k=100':>12}")
        print("-" * 80)

        # Accuracy metrics
        for metric in ['NDCG', 'Recall', 'Precision', 'MRR', 'Hit Rate']:
            row = f"{metric:<20}"
            for k in [10, 20, 50, 100]:
                key = f'val_{metric.lower().replace(" ", "_")}@{k}'
                value = result[key]
                row += f"{value:>12.4f}"
            print(row)

        print("-" * 80)

        # Coverage and distribution metrics
        for metric in ['Item Coverage', 'Gini Index', 'Simpson Index']:
            row = f"{metric:<20}"
            for k in [10, 20, 50, 100]:
                key = f'val_{metric.lower().replace(" ", "_")}@{k}'
                value = result[key]
                row += f"{value:>12.4f}"
            print(row)

        print("-" * 80)

        # NEW DIVERSITY METRICS
        diversity_metrics = [
            ('Simpson (Genre)', 'simpson_index_genre'),
            ('Intra-list Div', 'intra_list_diversity'),
            ('Genre Entropy', 'normalized_genre_entropy'),
            ('Unique Genres', 'unique_genres_count')
        ]

        for metric_name, metric_key in diversity_metrics:
            row = f"{metric_name:<20}"
            for k in [10, 20, 50, 100]:
                key = f'val_{metric_key}@{k}'
                value = result[key]
                row += f"{value:>12.4f}"
            print(row)

        print("-" * 80)

        # NEW POPULARITY METRICS
        popularity_metrics = [
            ('Popularity Lift', 'popularity_lift'),
            ('Pop. Calibration', 'popularity_calibration')
        ]

        for metric_name, metric_key in popularity_metrics:
            row = f"{metric_name:<20}"
            for k in [10, 20, 50, 100]:
                key = f'val_{metric_key}@{k}'
                value = result[key]
                row += f"{value:>12.4f}"
            print(row)


