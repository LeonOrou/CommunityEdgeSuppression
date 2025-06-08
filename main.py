import torch
import numpy as np
import os
from collections import defaultdict
import warnings
from config import Config
from evaluation import evaluate_model, evaluate_current_model_ndcg, print_metric_results
from models import calculate_bpr_loss, multivae_loss, get_model
import torch.optim as optim
from dataset import RecommendationDataset, sample_negative_items, prepare_adj_tensor, prepare_training_data
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

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.epochs // 5 if config.epochs >= 5 else 1,
        T_mult=1,
        eta_min=config.min_lr,
        last_epoch=-1
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
    # TODO: make big batches with 200k interactions maximum and have only this one big batch on GPU
    # TODO get smaller batches with batch_size = config.batch_size from this big batch
    if len(dataset.train_df) < 200000:
        all_train_users, all_train_items, train_indices = prepare_training_data(
            dataset.train_val_df, device=device)
    else:
        all_train_users, all_train_items, train_indices = prepare_training_data(
            dataset.train_df, device='cpu')

    num_train = len(train_indices)

    adj_tens = prepare_adj_tensor(dataset)

    model.train()

    for epoch in range(config.epochs):
        if config.use_dropout:
            edge_weights_modified = community_edge_suppression(adj_tens, config)
            if config.model_name == 'LightGCN':
                edge_weights_modified = torch.concatenate((edge_weights_modified, edge_weights_modified))
            current_edge_weight = edge_weights_modified.to(device)
        else:
            current_edge_weight = dataset.complete_edge_weight

        dataset.current_edge_weight = current_edge_weight

        total_loss = 0
        num_batches = 0

        perm = torch.randperm(num_train)
        train_indices_shuffled = train_indices[perm]

        # make big batches with 200k interactions maximum to run locally
        for start_idx in range(0, num_train, 200000):
            biggi_batch_indices = train_indices_shuffled[start_idx:start_idx + 200000]

            biggi_train_users = all_train_users[biggi_batch_indices].to(device)
            biggi_train_items = all_train_items[biggi_batch_indices].to(device)
            biggi_batch_indices = biggi_batch_indices.to(device)  # only have big batch on GPU

            for i in range(0, num_train, batch_size):
                batch_indices = biggi_batch_indices[i:i + batch_size]

                if len(batch_indices) == 0:
                    continue

                batch_users = biggi_train_users[batch_indices]
                batch_pos_items = biggi_train_items[batch_indices]

                batch_neg_items = sample_negative_items(
                    batch_users,
                    batch_pos_items,
                    dataset.num_items,
                    user_positive_items,
                    device)

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
                model=model, dataset=dataset, config=config,
                k_values=config.evaluate_top_k, stage='cv'
            )

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

    print("Training final model with complete graph structure...")

    model = get_model(config=config, dataset=dataset)
    # Final training: use ALL data in graph, but only train+val for loss
    final_model, final_edge_index, final_edge_weight = train_model(
        model=model, dataset=dataset, config=config, stage='full_train',
        fold_num=None,)

    test_metrics = evaluate_model(
        model=final_model, dataset=dataset, config=config,
        k_values=config.evaluate_top_k, stage='full_train')

    log_test_metrics_to_wandb(test_metrics, config)

    model_artifact = wandb.Artifact(
        name=f"model_{config.model_name}_{config.dataset_name}",
        type="model",
        description=f"Trained {config.model_name} model on {config.dataset_name} dataset")

    model_path = "final_model.pth"
    torch.save(final_model.state_dict(), model_path)
    model_artifact.add_file(model_path)

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

    print_metric_results(test_metrics, "FINAL TEST SET RESULTS")

    wandb.log({
        'experiment/total_folds': n_folds,
        'experiment/best_cv_ndcg@10': max([cv_results[i]['val_ndcg@10'] for i in range(len(cv_results))]),
        'experiment/final_test_ndcg@10': test_metrics[10]['ndcg'],
        # 'experiment/cv_std_ndcg@10': cv_df['val_ndcg@10'].std() if cv_results else 0,
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
        # Reconstruct metrics dict for this fold
        fold_metrics = {}
        for key, value in result.items():
            if key != 'fold' and '@' in key:
                # Extract k value and metric name from key like 'val_ndcg@10'
                metric_part, k_str = key.split('@')
                k = int(k_str)
                metric_name = metric_part.replace('val_', '')

                if k not in fold_metrics:
                    fold_metrics[k] = {}
                fold_metrics[k][metric_name] = value

        print_metric_results(fold_metrics, f"Fold {result['fold']}")

