import torch
import torch.nn.functional as F
import numpy as np
import time
import wandb
from collections import defaultdict
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sp


def train_multivae(model, dataset, config, optimizer, scheduler, device,
                   stage='cv', fold_num=None, verbose=True):
    """
    General MultiVAE training function that handles the complete training loop with sparse matrices.

    Args:
        model: The MultiVAE model to train
        dataset: Dataset object containing train/val/test data
        config: Configuration object with hyperparameters
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        device: Training device (cuda/cpu)
        stage: 'cv' for cross-validation, 'full_train' for final training
        fold_num: Current fold number for logging (None for final training)
        verbose: Whether to print training progress

    Returns:
        tuple: (trained_model, best_epoch, best_val_ndcg)
    """
    print(f"Starting MultiVAE training with sparse matrices...")
    print(f"Complete graph: {len(dataset.complete_df)} interactions")
    print(f"Training interactions: {len(dataset.train_df)} (used for loss)")
    print(f"Val interactions: {len(dataset.val_df)} (used for evaluation)")
    print(f"Test interactions: {len(dataset.test_df)} (used for evaluation)")

    # Training parameters
    batch_size = config.batch_size
    best_val_ndcg = 0
    patience = config.patience if hasattr(config, 'patience') else 50
    patience_counter = 0
    val_history = []
    best_epoch = 0
    best_model_state = None

    # Prepare sparse user-item matrices for MultiVAE
    train_sparse_matrix, val_sparse_matrix = _prepare_multivae_sparse_matrices(dataset)

    num_users = train_sparse_matrix.shape[0]
    model.train()

    # Main training loop
    for epoch in range(config.epochs):
        # TODO: make biggi batches
        epoch_loss, kl_weight = _train_multivae_epoch_sparse(
            model, train_sparse_matrix, optimizer, batch_size,
            epoch, config, device
        )

        # Evaluation and logging
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            val_ndcg = _evaluate_multivae_sparse(model, dataset, device)
            val_history.append(val_ndcg)
            current_lr = optimizer.param_groups[0]['lr']

            # Log metrics to WandB
            _log_multivae_training_metrics(
                epoch, epoch_loss, val_ndcg, current_lr, kl_weight,
                patience_counter, fold_num
            )

            if verbose:
                print(f'  Epoch {epoch + 1:3d}/{config.epochs}, Loss: {epoch_loss:.4f}, '
                      f'Val NDCG@10: {val_ndcg:.4f}, KL Weight: {kl_weight:.4f}')

            # Early stopping logic
            should_stop, best_val_ndcg, patience_counter, best_model_state, best_epoch = _check_multivae_early_stopping(
                val_ndcg, best_val_ndcg, patience_counter, patience, val_history,
                model, epoch, verbose
            )

            if should_stop:
                break

    # Load best model if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Log final training results
    _log_multivae_final_results(best_epoch, best_val_ndcg, epoch + 1,
                                patience_counter >= patience, fold_num)

    return model, best_epoch, best_val_ndcg


def _prepare_multivae_sparse_matrices(dataset):
    train_matrix = _create_sparse_user_item_matrix(
        dataset.train_df, dataset.num_users, dataset.num_items)

    val_matrix = _create_sparse_user_item_matrix(
        dataset.val_df, dataset.num_users, dataset.num_items)

    return train_matrix, val_matrix


def _create_sparse_user_item_matrix(data_df, num_users, num_items):
    users = data_df['user_encoded'].values
    items = data_df['item_encoded'].values
    ratings = data_df['rating'].values

    # Create sparse matrix
    matrix = csr_matrix(
        (ratings, (users, items)),
        shape=(num_users, num_items),
        dtype=np.float32)

    return matrix

def _train_multivae_epoch_sparse(model, train_sparse_matrix, optimizer, batch_size,
                                 epoch, config, device):
    """
    Train MultiVAE for one epoch using sparse matrices.

    Args:
        model: MultiVAE model
        train_sparse_matrix: Training sparse user-item matrix
        optimizer: PyTorch optimizer
        batch_size: Batch size
        epoch: Current epoch number
        config: Configuration object
        device: Training device

    Returns:
        tuple: (average_loss, kl_annealing_weight)
    """
    model.train()
    total_loss = 0
    num_batches = 0
    num_users = train_sparse_matrix.shape[0]

    # KL annealing schedule
    if hasattr(config, 'total_anneal_steps') and config.total_anneal_steps > 0:
        anneal_cap = getattr(config, 'anneal_cap', 0.2)
        kl_weight = min(anneal_cap, 1.0 * epoch / config.total_anneal_steps)
    else:
        kl_weight = getattr(config, 'kl_weight', 1.0)

    # Create user indices for batching
    user_indices = np.random.permutation(num_users)

    for start_idx in range(0, num_users, batch_size):
        end_idx = min(start_idx + batch_size, num_users)
        batch_user_indices = user_indices[start_idx:end_idx]

        # Convert sparse batch to dense tensor
        batch_data = _sparse_to_dense_batch(train_sparse_matrix, batch_user_indices, device)

        # Forward pass
        recon_batch, mu, logvar = model(batch_data)

        # Calculate loss
        loss = _multivae_loss_batch(recon_batch, batch_data, mu, logvar, kl_weight)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, kl_weight


def _multivae_loss_batch(recon_batch, rating_batch, mu, logvar, anneal=1.0):
    """
    Calculate MultiVAE loss for a batch.

    Args:
        recon_batch: Reconstructed ratings
        rating_batch: True ratings
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        anneal: KL annealing weight

    Returns:
        torch.Tensor: Total loss
    """
    # Reconstruction loss (negative log-likelihood)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_batch, 1) * rating_batch, dim=1))

    # KL divergence
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD


def _get_sparse_user_data(sparse_matrix, user_id, device):
    """
    Get a single user's data from sparse matrix as dense tensor.

    Args:
        sparse_matrix: Sparse user-item matrix
        user_id: User ID
        device: Target device

    Returns:
        torch.Tensor: Dense user vector
    """
    user_row = sparse_matrix.getrow(user_id).toarray().flatten()
    return torch.tensor(user_row, device=device, dtype=torch.float32).unsqueeze(0)


def _log_multivae_training_metrics(epoch, epoch_loss, val_ndcg, current_lr, kl_weight,
                                   patience_counter, fold_num):
    """Log MultiVAE training metrics to WandB."""
    log_dict = {
        'epoch': epoch,
        'train_loss': epoch_loss,
        'val_ndcg@10': val_ndcg,
        'learning_rate': current_lr,
        'kl_annealing_weight': kl_weight,
        'patience_counter': patience_counter
    }

    # Add fold-specific prefix if in cross-validation
    if fold_num is not None:
        log_dict = {f'fold_{fold_num}/{k}': v for k, v in log_dict.items()}
    else:
        log_dict = {f'final_training/{k}': v for k, v in log_dict.items()}

    wandb.log(log_dict)


def _check_multivae_early_stopping(val_ndcg, best_val_ndcg, patience_counter, patience,
                                   val_history, model, epoch, verbose):
    """Check early stopping conditions for MultiVAE and update best model."""
    best_model_state = None
    best_epoch = epoch

    # Check for improvement
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

    # Check if learning has plateaued
    if len(val_history) >= 5:
        recent_std = np.std(val_history[-5:])
        if recent_std < 0.0001 and patience_counter >= patience // 2:
            if verbose:
                print(f"  Early stopping at epoch {epoch} - validation plateaued")
            return True, best_val_ndcg, patience_counter, best_model_state, best_epoch

    # Check patience limit
    if patience_counter >= patience:
        if verbose:
            print(f"  Early stopping at epoch {epoch} - no improvement for {patience} checks")
            print(f"  Best epoch was {best_epoch} with NDCG@10: {best_val_ndcg:.4f}")
        return True, best_val_ndcg, patience_counter, best_model_state, best_epoch

    return False, best_val_ndcg, patience_counter, best_model_state, best_epoch


def _log_multivae_final_results(best_epoch, best_val_ndcg, total_epochs, early_stopped, fold_num):
    """Log final MultiVAE training results to WandB."""
    final_log_dict = {
        'best_epoch': best_epoch,
        'best_val_ndcg@10': best_val_ndcg,
        'total_epochs': total_epochs,
        'early_stopped': early_stopped
    }

    if fold_num is not None:
        final_log_dict = {f'fold_{fold_num}/final_{k}': v for k, v in final_log_dict.items()}
    else:
        final_log_dict = {f'final_training/final_{k}': v for k, v in final_log_dict.items()}

    wandb.log(final_log_dict)


def train_model_multivae(dataset, model, config, stage='cv', fold_num=None, verbose=True):
    """
    Main training function for MultiVAE that sets up the training environment and calls train_multivae.

    Args:
        dataset: Dataset object
        model: MultiVAE model to train
        config: Configuration object
        stage: 'cv' for cross-validation, 'full_train' for final training
        fold_num: Current fold number for logging (None for final training)
        verbose: Whether to print training progress

    Returns:
        tuple: (trained_model, best_epoch, best_val_ndcg)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training MultiVAE on device: {device}")

    # Move model to device
    model = model.to(device)

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=getattr(config, 'weight_decay', 0.0)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.epochs // 5 if config.epochs >= 5 else 1,
        T_mult=1,
        eta_min=getattr(config, 'min_lr', 1e-6),
        last_epoch=-1
    )

    # Adjust dataset for final training stage
    if stage == 'full_train':
        dataset.train_df = dataset.train_val_df  # Use all training+validation data
        dataset.val_df = dataset.test_df  # Use test set as validation

    # Train the model using the general training function
    trained_model, best_epoch, best_val_ndcg = train_multivae(
        model=model,
        dataset=dataset,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        stage=stage,
        fold_num=fold_num,
        verbose=verbose
    )

    return trained_model, best_epoch, best_val_ndcg