import time
from collections import defaultdict

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import wandb
from wandb_logging import log_metrics_to_wandb
from evaluation import evaluate_current_model_ndcg, evaluate_current_model_ndcg_vectorized
from dataset import RecommendationDataset, sample_negative_items, prepare_adj_tensor, prepare_training_data
from models import calculate_bpr_loss
from utils_functions import community_edge_suppression
from scipy.sparse import csr_matrix
import torch
import torch.optim as optim
import numpy as np


def train_itemknn(model, dataset, config, stage='cv', fold_num=None, verbose=True):
    """
    General ItemKNN training function that handles the complete training process.

    Args:
        model: The ItemKNN model to train
        dataset: Dataset object containing train/val/test data
        config: Configuration object with hyperparameters
        stage: 'cv' for cross-validation, 'full_train' for final training
        fold_num: Current fold number for logging (None for final training)
        verbose: Whether to print training progress

    Returns:
        trained_model
    """
    print(f"Starting ItemKNN training...")

    training_interactions = dataset.train_df[['user_encoded', 'item_encoded', 'rating']].values

    start_time = time.time()

    # TODO: make multiple training iterations to proof community edge suppression as one-shot learning
    if config.use_dropout:
        current_edge_weights = community_edge_suppression(
            torch.tensor(training_interactions, device=config.device), config).cpu().numpy()
        training_interactions[:, 2] = current_edge_weights

    model.fit(training_interactions)

    training_time = time.time() - start_time

    # Validate the model if validation data is available
    validation_ndcg = None
    if len(dataset.val_df) > 0:
        from evaluation import evaluate_current_model_ndcg
        validation_ndcg = evaluate_current_model_ndcg(model, dataset, model_type='ItemKNN', k=10)

    # Log training results
    _log_itemknn_training_results(
        model, training_time, validation_ndcg, fold_num, verbose
    )

    return model


def _log_itemknn_training_results(model, training_time, validation_ndcg, fold_num, verbose):
    """Log ItemKNN training completion and results."""
    # Calculate model statistics
    sparsity = _calculate_similarity_sparsity(model.similarity_matrix)
    avg_neighbors = _calculate_avg_neighbors(model.similarity_matrix)

    log_dict = {
        'training_time': training_time,
        'similarity_sparsity': sparsity,
        'avg_neighbors_per_item': avg_neighbors,
        'training_completed': True
    }

    if validation_ndcg is not None:
        log_dict['val_ndcg@10'] = validation_ndcg

    # Add fold-specific prefix if in cross-validation
    if fold_num is not None:
        log_dict = {f'fold_{fold_num}/itemknn_{k}': v for k, v in log_dict.items()}
    else:
        log_dict = {f'final_training/itemknn_{k}': v for k, v in log_dict.items()}

    wandb.log(log_dict)

    if verbose:
        print(f"  Training completed in {training_time:.2f} seconds")
        print(f"  Similarity matrix sparsity: {sparsity:.4f}")
        print(f"  Average neighbors per item: {avg_neighbors:.1f}")
        if validation_ndcg is not None:
            print(f"  Validation NDCG@10: {validation_ndcg:.4f}")


def _calculate_similarity_sparsity(similarity_matrix):
    """Calculate sparsity of the similarity matrix."""
    total_elements = similarity_matrix.shape[0] * similarity_matrix.shape[1]
    non_zero_elements = similarity_matrix.nnz
    sparsity = 1 - (non_zero_elements / total_elements)
    return sparsity

def _prepare_multivae_sparse_matrices(dataset):
    train_matrix = _create_sparse_user_item_matrix(
        dataset.train_df, dataset.num_users, dataset.num_items)

    val_matrix = _create_sparse_user_item_matrix(
        dataset.val_df, dataset.num_users, dataset.num_items)

    return train_matrix, val_matrix

def _create_sparse_user_item_matrix(data_df, num_users, num_items):
    matrix = csr_matrix(
        (data_df['rating'].values, (data_df['user_encoded'].values, data_df['item_encoded'].values)),
        shape=(num_users, num_items),
        dtype=np.float32)
    return matrix


def _sparse_to_dense_batch(sparse_matrix, user_indices, device):
    return torch.tensor(sparse_matrix[user_indices].toarray(), device=device, dtype=torch.float32)


def multivae_loss(recon_batch, rating_weights, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(torch.nn.functional.log_softmax(recon_batch, 1) * rating_weights, dim=1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD


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
    anneal_cap = getattr(config, 'anneal_cap', 0.4)
    kl_weight = min(anneal_cap, 1.0 * epoch / config.total_anneal_steps)

    # Create user indices for batching
    user_indices = np.random.permutation(num_users)

    for start_idx in range(0, num_users, batch_size):
        end_idx = min(start_idx + batch_size, num_users)
        batch_user_indices = user_indices[start_idx:end_idx]

        # Convert sparse batch to dense tensor
        batch_data = _sparse_to_dense_batch(train_sparse_matrix, batch_user_indices, device)

        # Forward pass
        recon_batch, mu, logvar = model(batch_data)

        loss = multivae_loss(recon_batch, batch_data, mu, logvar, kl_weight)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, kl_weight


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


def train_multivae(model, dataset, config, optimizer, device,
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
        tuple: trained_model
    """
    print(f"Starting MultiVAE...")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 'max' if monitoring recall/ndcg
        factor=0.8,  # Reduce learning rate by this factor
        patience=10,  # Wait epochs before reducing
        min_lr=1e-5,  # Don't go below this
        threshold=0.005  # Minimum improvement to count
    )

    batch_size = config.batch_size
    best_val_ndcg = 0
    patience = config.patience if hasattr(config, 'patience') else 15
    patience_counter = 0
    val_history = []
    best_epoch = 0
    best_model_state = None

    # Prepare sparse user-item matrices for MultiVAE
    train_sparse_matrix, val_sparse_matrix = _prepare_multivae_sparse_matrices(dataset)

    num_users = train_sparse_matrix.shape[0]
    model.train()

    for epoch in range(config.epochs):
        if config.use_dropout:
            current_edge_weight = community_edge_suppression(torch.tensor(dataset.train_df.values, device=device), config).cpu().numpy()

            train_sparse_matrix = csr_matrix(
                (current_edge_weight, (dataset.train_df['user_encoded'].values, dataset.train_df['item_encoded'].values)),
                shape=(num_users, dataset.num_items),
                dtype=np.float32)
        epoch_loss, kl_weight = _train_multivae_epoch_sparse(
            model, train_sparse_matrix, optimizer, batch_size,
            epoch, config, device)

        val_ndcg = evaluate_current_model_ndcg(model, dataset, model_type='MultiVAE', k=10)
        scheduler.step(val_ndcg)

        # Evaluation and logging
        if epoch == 0 or (epoch+1) % 10 == 0 or epoch == config.epochs - 1:

            val_history.append(val_ndcg)
            current_lr = optimizer.param_groups[0]['lr']

            if verbose:
                print(f'  Epoch {epoch + 1:3d}/{config.epochs}, Loss: {epoch_loss:.4f}, '
                      f'current LR: {current_lr:.6f}, Val NDCG@10: {val_ndcg:.4f}')

            # Early stopping logic
            should_stop, best_val_ndcg, patience_counter, best_model_state, best_epoch = _check_multivae_early_stopping(
                val_ndcg, best_val_ndcg, patience_counter, patience, val_history,
                model, epoch, verbose)

            if should_stop:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def train_lightgcn(model, dataset, config, optimizer, user_positive_items,
                   device, stage='cv', fold_num=None, verbose=True):
    """
    General LightGCN training function that handles the complete training loop.

    Args:
        model: The LightGCN model to train
        dataset: Dataset object containing train/val/test data
        config: Configuration object with hyperparameters
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        user_positive_items: Dict mapping users to their positive items
        device: Training device (cuda/cpu)
        stage: 'cv' for cross-validation, 'full_train' for final training
        fold_num: Current fold number for logging (None for final training)
        verbose: Whether to print training progress

    Returns:
        trained_model
    """
    print(f"Starting LightGCN training...")

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,  # First restart after 15 epochs
        T_mult=2,  # Double cycle length after each restart
        eta_min=1e-5  # Minimum learning rate
    )

    # Training parameters
    batch_size = config.batch_size
    best_val_ndcg = 0
    patience = config.patience if hasattr(config, 'patience') else 50
    patience_counter = 0
    val_history = []
    best_epoch = 0
    best_model_state = None

    # Pre-convert training data to GPU or CPU based on size
    if len(dataset.train_df) < 200000:
        all_train_users, all_train_items, train_indices = prepare_training_data(
            dataset.train_df, device=device)
    else:
        all_train_users, all_train_items, train_indices = prepare_training_data(
            dataset.train_df, device='cpu')

    num_train = len(train_indices)
    adj_tens = prepare_adj_tensor(dataset)
    model.train()

    for epoch in range(config.epochs):
        if config.use_dropout:
            edge_weights_modified = community_edge_suppression(adj_tens, config)
            # bidirectional edges for LightGCN
            edge_weights_modified = torch.concatenate((edge_weights_modified, edge_weights_modified))
            current_edge_weight = edge_weights_modified.to(device)
        else:
            current_edge_weight = dataset.complete_edge_weight

        dataset.current_edge_weight = current_edge_weight

        # Training step
        epoch_loss = _train_lightgcn_epoch(
            model, dataset, config, optimizer, all_train_users, all_train_items,
            train_indices, user_positive_items, batch_size, device
        )
        scheduler.step()

        # Evaluation and logging
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:

            # val_ndcg = evaluate_current_model_ndcg_vectorized(model, dataset, k=10)
            val_ndcg = evaluate_current_model_ndcg(model, dataset, k=10)

            val_history.append(val_ndcg)
            current_lr = optimizer.param_groups[0]['lr']

            if verbose:
                suppression_status = "ON" if config.use_dropout else "OFF"
                print(f'  Epoch {epoch+1:3d}/{config.epochs}, Loss: {epoch_loss:.4f}, current LR: {current_lr:.6f}, '
                      f'Val NDCG@10: {val_ndcg:.4f} (Suppression: {suppression_status})')

            # Early stopping logic
            should_stop, best_val_ndcg, patience_counter, best_model_state, best_epoch = _check_early_stopping(
                val_ndcg, best_val_ndcg, patience_counter, patience, val_history,
                model, epoch, verbose
            )

            if should_stop:
                break

    # Load best model if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def _train_lightgcn_epoch(model, dataset, config, optimizer, all_train_users, all_train_items,
                          train_indices, user_positive_items, batch_size, device):
    """Train for one epoch and return average loss."""
    total_loss = 0
    num_batches = 0
    num_train = len(train_indices)

    # Shuffle training indices
    perm = torch.randperm(num_train)
    train_indices_shuffled = train_indices[perm]

    # Process in big batches of 200k to manage memory
    for start_idx in range(0, num_train, 200000):
        biggi_batch_indices = train_indices_shuffled[start_idx:start_idx + 200000]
        biggi_train_users = all_train_users[biggi_batch_indices].to(device)
        biggi_train_items = all_train_items[biggi_batch_indices].to(device)
        biggi_batch_indices = biggi_batch_indices.to(device)

        # Process smaller batches within the big batch
        for i in range(0, len(biggi_batch_indices), batch_size):
            batch_indices = biggi_batch_indices[i:i + batch_size]
            if len(batch_indices) == 0:
                continue

            batch_users = biggi_train_users[i:i + batch_size]
            batch_pos_items = biggi_train_items[i:i + batch_size]

            batch_neg_items = dataset.sample_negative_items(user_ids=batch_users)

            # Forward pass
            user_emb, item_emb = model(dataset.complete_edge_index, dataset.current_edge_weight)
            batch_user_emb = user_emb[batch_users.long()]
            batch_pos_item_emb = item_emb[batch_pos_items.long()]
            batch_neg_item_emb = item_emb[batch_neg_items.long()]

            # Calculate loss
            bpr_loss = calculate_bpr_loss(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)

            # Add L2 regularization
            l2_reg = config.reg * (
                    batch_user_emb.norm(2).pow(2) +
                    batch_pos_item_emb.norm(2).pow(2) +
                    batch_neg_item_emb.norm(2).pow(2)
            ) / batch_size

            loss = bpr_loss + l2_reg

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def _check_early_stopping(val_ndcg, best_val_ndcg, patience_counter, patience,
                          val_history, model, epoch, verbose):
    """Check early stopping conditions and update best model."""
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


def _calculate_avg_neighbors(similarity_matrix):
    """Calculate average number of neighbors per item."""
    neighbors_per_item = np.array((similarity_matrix > 0).sum(axis=1)).flatten()
    return neighbors_per_item.mean()


def train_model_itemknn(dataset, model, config, stage='cv', fold_num=None, verbose=True):
    """
    Main training function for ItemKNN that sets up the training environment and calls train_itemknn.

    Args:
        dataset: Dataset object
        model: ItemKNN model to train
        config: Configuration object
        stage: 'cv' for cross-validation, 'full_train' for final training
        fold_num: Current fold number for logging (None for final training)
        verbose: Whether to print training progress

    Returns:
        trained_model
    """
    print(f"Training ItemKNN model...")

    # Adjust dataset for final training stage
    if stage == 'full_train':
        dataset.train_df = dataset.train_val_df  # Use all training+validation data
        dataset.val_df = dataset.test_df  # Use test set as validation

    # Train the model using the general training function
    trained_model = train_itemknn(
        model=model,
        dataset=dataset,
        config=config,
        stage=stage,
        fold_num=fold_num,
        verbose=verbose
    )

    return trained_model


def train_model(dataset, model, config, stage='cv', fold_num=None, verbose=True):
    """
    Main training function that sets up the training environment and calls train_lightgcn.

    Args:
        dataset: Dataset object
        model: Model to train
        config: Configuration object
        stage: 'cv' for cross-validation, 'full_train' for final training
        fold_num: Current fold number for logging (None for final training)
        verbose: Whether to print training progress

    Returns:
        trained_model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    if stage == 'full_train':
        dataset.train_df = dataset.train_val_df  # Use all training+validation data
        dataset.val_df = dataset.test_df  # Use test set as validation

    if config.model_name == 'ItemKNN':
        trained_model = train_itemknn(
            model=model,
            dataset=dataset,
            config=config,
            stage=stage,
            fold_num=fold_num,
            verbose=verbose
        )
        return trained_model

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate,
                                 weight_decay=getattr(config, 'weight_decay', 0.0))

    if config.model_name == 'MultiVAE':
        trained_model = train_multivae(
            model=model,
            dataset=dataset,
            config=config,
            optimizer=optimizer,
            device=device,
            stage=stage,
            fold_num=fold_num,
            verbose=verbose
        )
        return trained_model

    elif config.model_name == 'LightGCN':
        dataset.complete_edge_index = dataset.complete_edge_index.to(device)
        dataset.complete_edge_weight = dataset.complete_edge_weight.to(device)

        # Create user positive items mapping for better negative sampling
        user_positive_items = defaultdict(set)
        for user_id, group in dataset.complete_df.groupby('user_encoded')['item_encoded']:
            user_positive_items[user_id] = set(group.values)

        trained_model = train_lightgcn(
            model=model,
            dataset=dataset,
            config=config,
            optimizer=optimizer,
            user_positive_items=user_positive_items,
            device=device,
            stage=stage,
            fold_num=fold_num,
            verbose=verbose)

        return trained_model



