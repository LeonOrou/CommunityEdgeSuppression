import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import ndcg_score
import itertools
import time
from collections import defaultdict
import random
from scipy.sparse import csr_matrix, coo_matrix
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models import LightGCN, ItemKNN, MultiVAE, calculate_bpr_loss, multivae_loss
from dataset import RecommendationDataset
import scipy as sp

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def create_sparse_matrices(train_df, test_df, num_users, num_items):
    """Create sparse matrices for efficient computation"""
    # Training matrix
    train_matrix = csr_matrix(
        (np.ones(len(train_df)), (train_df['user_encoded'], train_df['item_encoded'])),
        shape=(num_users, num_items), dtype=np.float32
    )

    # Test matrix
    test_matrix = csr_matrix(
        (np.ones(len(test_df)), (test_df['user_encoded'], test_df['item_encoded'])),
        shape=(num_users, num_items), dtype=np.float32
    )

    return train_matrix, test_matrix


def calculate_ndcg(y_true, y_scores, k=10):
    """Calculate NDCG@k_values"""
    if len(y_true) == 0:
        return 0.0

    # Create relevance scores (1 for relevant, 0 for non-relevant)
    relevance_scores = np.zeros(len(y_scores))
    relevance_scores[y_true] = 1

    # Get top-k_values predictions
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    top_k_relevance = relevance_scores[top_k_indices]

    if np.sum(top_k_relevance) == 0:
        return 0.0

    # Calculate DCG@k_values
    dcg = np.sum((2 ** top_k_relevance - 1) / np.log2(np.arange(2, k + 2)))

    # Calculate IDCG@k_values
    ideal_relevance = np.sort(relevance_scores)[::-1][:k]
    idcg = np.sum((2 ** ideal_relevance - 1) / np.log2(np.arange(2, k + 2)))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_current_model_ndcg(model, dataset, k=10):
    """
    only get ndcg for early stopping, minimum calculations
    only necessary steps to calculate ndcg = calculate_ndcg(true_items, scores_filtered, k) for val_df
    """
    model.eval()

    ndcg_scores = []

    # Create a set of training interactions for each user to exclude from evaluation
    train_user_items = dataset.train_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

    with torch.no_grad():
        user_emb, item_emb = model(dataset.complete_edge_index, dataset.complete_edge_weight)

        user_val_items = dataset.val_df.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

        for user_id, true_items in user_val_items.items():
            if user_id >= dataset.num_users:
                continue

            user_embedding = user_emb[user_id:user_id + 1]
            scores = torch.matmul(user_embedding, item_emb.T).squeeze().cpu().numpy()

            # Exclude items that the user interacted with during training
            train_items = list(train_user_items[user_id])
            scores_filtered = scores.copy()
            scores_filtered[train_items] = float('-inf')

            ndcg = calculate_ndcg(true_items, scores_filtered, k)
            ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def vectorized_ndcg_at_k(y_true_matrix, y_pred_matrix, k=10):
    """
    Vectorized NDCG@K calculation for all users at once

    Args:
        y_true_matrix: sparse matrix (num_users x num_items) with ground truth
        y_pred_matrix: dense matrix (num_users x num_items) with predictions
        k: top-k for NDCG calculation

    Returns:
        ndcg_scores: array of NDCG scores for each user
    """
    num_users, num_items = y_pred_matrix.shape
    ndcg_scores = np.zeros(num_users)

    # Convert sparse to dense for ground truth
    y_true_dense = y_true_matrix.toarray()

    # Get top-k predictions for all users at once
    top_k_indices = np.argpartition(-y_pred_matrix, k - 1, axis=1)[:, :k]

    # Vectorized NDCG calculation
    for user_idx in range(num_users):
        if y_true_matrix[user_idx].nnz == 0:  # No ground truth items
            continue

        # Get top-k items for this user
        user_top_k = top_k_indices[user_idx]
        user_pred_scores = y_pred_matrix[user_idx, user_top_k]

        # Sort by prediction scores (descending)
        sorted_indices = np.argsort(-user_pred_scores)
        sorted_items = user_top_k[sorted_indices]

        # Get relevance scores (binary: 1 if in ground truth, 0 otherwise)
        relevance = y_true_dense[user_idx, sorted_items]

        # Calculate DCG
        dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))

        # Calculate IDCG (ideal DCG)
        ideal_relevance = np.sort(y_true_dense[user_idx])[::-1][:k]
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))

        # Calculate NDCG
        if idcg > 0:
            ndcg_scores[user_idx] = dcg / idcg

    # Return mean NDCG for users with ground truth items
    valid_users = y_true_matrix.getnnz(axis=1) > 0
    return np.mean(ndcg_scores[valid_users]) if np.any(valid_users) else 0.0


def evaluate_model_vectorized(model, train_matrix, test_matrix, model_type, device, dataset=None, k=10):
    """Vectorized model evaluation"""
    if model_type != 'ItemKNN':
        model.eval()
    num_users, num_items = train_matrix.shape

    with torch.no_grad():
        if model_type == 'ItemKNN':
            # Batch prediction for all users
            predictions = np.zeros((num_users, num_items))

            # Process users in batches to manage memory
            batch_size = 1000
            for start_idx in range(0, num_users, batch_size):
                end_idx = min(start_idx + batch_size, num_users)

                for user_idx in range(start_idx, end_idx):
                    if train_matrix[user_idx].nnz > 0:  # User has training interactions
                        preds = model.predict(user_idx)[0]  # 0 as we only take one user always
                        item_ids = np.array(preds[:, 0], dtype=np.int64)  # Get item IDs
                        scores = preds[:, 1]  # Get predicted scores
                        predictions[user_idx, item_ids] = scores

            # Mask training items (set to -inf)
            train_mask = train_matrix.toarray() > 0
            predictions[train_mask] = -np.inf

        elif model_type == 'LightGCN':
            # Use the optimized LightGCN evaluation
            if dataset is not None:
                return evaluate_current_model_ndcg(model, dataset, k)
            else:
                # Fallback to original method if dataset not provided
                user_embs = model.user_embedding.weight.detach()  # (num_users, emb_dim)
                item_embs = model.item_embedding.weight.detach()  # (num_items, emb_dim)

                # Batch matrix multiplication for all predictions
                predictions = torch.mm(user_embs, item_embs.t()).cpu().numpy()  # (num_users, num_items)

                # Mask training items
                train_mask = train_matrix.toarray() > 0
                predictions[train_mask] = -np.inf

        elif model_type == 'MultiVAE':
            # Batch process all users
            train_tensor = torch.FloatTensor(train_matrix.toarray()).to(device)

            # Process in batches to manage GPU memory
            predictions = np.zeros((num_users, num_items))
            batch_size = 500

            for start_idx in range(0, num_users, batch_size):
                end_idx = min(start_idx + batch_size, num_users)
                batch_users = train_tensor[start_idx:end_idx]

                # Forward pass for batch
                recon_batch, _, _ = model(batch_users)
                batch_predictions = recon_batch.detach().cpu().numpy()

                predictions[start_idx:end_idx] = batch_predictions

            # Mask training items
            train_mask = train_matrix.toarray() > 0
            predictions[train_mask] = -np.inf

    # For LightGCN, we already returned the NDCG score above
    if model_type == 'LightGCN' and dataset is not None:
        return evaluate_current_model_ndcg(model, dataset, k)

    # Calculate NDCG using vectorized function for other models
    return vectorized_ndcg_at_k(test_matrix, predictions, k)


def create_edge_index(df, num_users):
    """Create edge index for GNN models"""
    users = df['user_encoded'].values
    items = df['item_encoded'].values + num_users  # Offset items

    # Create bidirectional edges
    edge_index = torch.stack([
        torch.tensor(np.concatenate([users, items]), dtype=torch.long),
        torch.tensor(np.concatenate([items, users]), dtype=torch.long)
    ], dim=0)

    return edge_index


def train_lightgcn_efficient(model, dataset, device, epochs=30, lr=0.005, batch_size=1024):
    """Fixed LightGCN training - compute embeddings inside each batch"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,  # First restart after 15 epochs
        T_mult=2,  # Double cycle length after each restart
        eta_min=1e-4  # Minimum learning rate
    )

    # Create edge index once
    edge_index = create_edge_index(dataset.train_df, dataset.num_users).to(device)

    # Prepare training data
    users = torch.tensor(dataset.train_df['user_encoded'].values, dtype=torch.long)
    pos_items = torch.tensor(dataset.train_df['item_encoded'].values, dtype=torch.long)

    # Create dataset and dataloader for efficient batching
    train_dataset = torch.utils.data.TensorDataset(users, pos_items)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_users, batch_pos_items in dataloader:
            batch_users = batch_users.to(device)
            batch_pos_items = batch_pos_items.to(device)

            # Vectorized negative sampling
            batch_neg_items = torch.randint(0, model.num_items, (len(batch_users),), device=device)

            # FIXED: Compute embeddings inside the batch loop
            user_emb, item_emb = model(edge_index)

            # Get embeddings for batch
            user_batch_emb = user_emb[batch_users]
            pos_item_batch_emb = item_emb[batch_pos_items]
            neg_item_batch_emb = item_emb[batch_neg_items]

            # Calculate BPR loss
            loss = calculate_bpr_loss(user_batch_emb, pos_item_batch_emb, neg_item_batch_emb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        val_ndcg = evaluate_current_model_ndcg(model, dataset, k=10)
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / num_batches:.4f}, ndcg@10: {val_ndcg:.4f}")


def train_multivae_efficient(model, train_matrix, test_matrix, device, epochs=80, lr=0.0005, batch_size=500, anneal_cap=0.2):
    """Efficient MultiVAE training with sparse matrix input"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 'max' if monitoring recall/ndcg
        factor=0.8,  # Reduce LR by half
        patience=15,  # Wait 10 epochs before reducing
        min_lr=1e-5,  # Don't go below this
        threshold=0.001  # Minimum improvement to count
    )

    # Convert sparse matrix to dense tensor (only once)
    train_tensor = torch.FloatTensor(train_matrix.toarray())
    train_dataset = torch.utils.data.TensorDataset(train_tensor)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # Annealing factor
        anneal = min(anneal_cap, 1.0 * epoch / 200)

        for batch_data, in dataloader:
            batch_data = batch_data.to(device)

            # Forward pass
            recon_batch, mu, logvar = model(batch_data)

            # Calculate loss
            loss = multivae_loss(recon_batch, batch_data, mu, logvar, anneal)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        val_ndcg = evaluate_model_vectorized(model, train_matrix, test_matrix, 'MultiVAE', device)
        scheduler.step(val_ndcg)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / num_batches:.4f}, ndcg@10: {val_ndcg:.4f}")


def hyperparameter_search():
    """Efficient hyperparameter search for all models using RecommendationDataset"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--dataset", type=str, default="ml-100k", help="Dataset name")
    parser.add_argument("--models", type=str, nargs='+', default=['MultiVAE'],
                        help="Models to search: LightGCN, ItemKNN, MultiVAE")
    # LightGCN
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    # ItemKNN
    parser.add_argument("--item_knn_topk", type=int, default=250)
    parser.add_argument("--shrink", type=int, default=10)
    # MultiVAE
    parser.add_argument("--hidden_dimension", type=int, default=800)
    parser.add_argument("--anneal_cap", type=float, default=0.4)

    args = parser.parse_args()

    # Generate unique run ID based on timestamp and dataset
    import datetime
    run_id = f"{args.dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Run ID: {run_id}")

    # Load dataset using RecommendationDataset
    print("Loading dataset...")
    if args.dataset == "LFM1M":
        dataset = RecommendationDataset(
            name='lfm',
            data_path='dataset/LFM1M/preprocessed',
            min_interactions=5
        )
    else:
        dataset = RecommendationDataset(
            name=args.dataset.lower(),
            min_interactions=5
        )

    # Load and prepare data
    dataset.load_data()
    dataset.prepare_data()

    # Use fold 0 for hyperparameter search
    dataset.train_df = dataset.train_val_df
    dataset.val_df = dataset.test_df

    # Create sparse matrices once
    train_matrix, test_matrix = create_sparse_matrices(
        dataset.train_df, dataset.val_df, dataset.num_users, dataset.num_items
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset: {dataset.num_users} users, {dataset.num_items} items")
    print(f"Train: {len(dataset.train_df)} interactions, Val: {len(dataset.val_df)} interactions")

    # Initialize results dictionaries for each model
    all_results = {model: [] for model in args.models}
    best_results = {}

    # Define hyperparameter spaces
    hyperparams = {
        'LightGCN': {
            'embedding_dim': [128],
            'n_layers': [3],
            'batch_size': [512]
        },
        'ItemKNN': {
            'topk': [125],
            'shrink': [50]
        },
        'MultiVAE': {
            'hidden_dimension': [800],
            'latent_dimension': [200],
            'anneal_cap': [0.4],
            'batch_size': [2048]  # Larger batches for efficiency
        }
    }

    # Search LightGCN
    if 'LightGCN' in args.models:
        print("\n=== Searching LightGCN Hyperparameters ===")
        best_lightgcn_ndcg = 0
        best_lightgcn_params = {}

        for emb_dim, n_layers, batch_size in itertools.product(
                hyperparams['LightGCN']['embedding_dim'],
                hyperparams['LightGCN']['n_layers'],
                hyperparams['LightGCN']['batch_size']
        ):
            print(f"Testing LightGCN: emb_dim={emb_dim}, n_layers={n_layers}, batch_size={batch_size}")

            model = LightGCN(
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                embedding_dim=emb_dim,
                num_layers=n_layers
            ).to(device)

            start_time = time.time()
            train_lightgcn_efficient(model, dataset, device, epochs=80, batch_size=batch_size)
            training_time = time.time() - start_time

            ndcg = evaluate_model_vectorized(model, train_matrix, test_matrix, 'LightGCN', device, dataset)

            result = {
                'model': 'LightGCN',
                'embedding_dim': emb_dim,
                'n_layers': n_layers,
                'batch_size': batch_size,
                'ndcg@10': ndcg,
                'training_time': training_time,
                'run_id': run_id
            }
            all_results['LightGCN'].append(result)

            print(f"NDCG@10: {ndcg:.4f}, Time: {training_time:.2f}s")

            if ndcg > best_lightgcn_ndcg:
                best_lightgcn_ndcg = ndcg
                best_lightgcn_params = result

            # Clear GPU memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        best_results['LightGCN'] = best_lightgcn_params

        # Save LightGCN results immediately
        save_model_results('LightGCN', all_results['LightGCN'], best_lightgcn_params, run_id, dataset)

    # Search ItemKNN
    if 'ItemKNN' in args.models:
        print("\n=== Searching ItemKNN Hyperparameters ===")
        best_itemknn_ndcg = 0
        best_itemknn_params = {}

        for topk, shrink in itertools.product(
                hyperparams['ItemKNN']['topk'],
                hyperparams['ItemKNN']['shrink']
        ):
            print(f"Testing ItemKNN: topk={topk}, shrink={shrink}")

            model = ItemKNN(
                num_items=dataset.num_items,
                num_users=dataset.num_users,
                k=topk,
                shrink=shrink
            )

            start_time = time.time()
            # Prepare interactions for ItemKNN

            model.fit(dataset.train_df[['user_encoded', 'item_encoded', 'rating']].values)
            training_time = time.time() - start_time

            ndcg = evaluate_model_vectorized(model, train_matrix, test_matrix, 'ItemKNN', device)

            result = {
                'model': 'ItemKNN',
                'topk': topk,
                'shrink': shrink,
                'ndcg@10': ndcg,
                'training_time': training_time,
                'run_id': run_id
            }
            all_results['ItemKNN'].append(result)

            print(f"NDCG@10: {ndcg:.4f}, Time: {training_time:.2f}s")

            if ndcg > best_itemknn_ndcg:
                best_itemknn_ndcg = ndcg
                best_itemknn_params = result

        best_results['ItemKNN'] = best_itemknn_params

        # Save ItemKNN results immediately
        save_model_results('ItemKNN', all_results['ItemKNN'], best_itemknn_params, run_id, dataset)

    # Search MultiVAE
    if 'MultiVAE' in args.models:
        print("\n=== Searching MultiVAE Hyperparameters ===")
        best_multivae_ndcg = 0
        best_multivae_params = {}

        for hidden_dim, latent_dim, anneal_cap, batch_size in itertools.product(
                hyperparams['MultiVAE']['hidden_dimension'],
                hyperparams['MultiVAE']['latent_dimension'],
                hyperparams['MultiVAE']['anneal_cap'],
                hyperparams['MultiVAE']['batch_size'],
        ):
            print(f"Testing MultiVAE: hidden_dim={hidden_dim}, latent_dim={latent_dim}, anneal_cap={anneal_cap}, batch_size={batch_size}")

            model = MultiVAE(
                p_dims=[latent_dim, hidden_dim, dataset.num_items],  # latent_dim=200
                dropout=0.5
            ).to(device)

            start_time = time.time()
            train_multivae_efficient(model, train_matrix, test_matrix, device, epochs=200, batch_size=batch_size,
                                     anneal_cap=anneal_cap)
            training_time = time.time() - start_time

            ndcg = evaluate_model_vectorized(model, train_matrix, test_matrix, 'MultiVAE', device)

            result = {
                'model': 'MultiVAE',
                'hidden_dimension': hidden_dim,
                'latent_dimension': latent_dim,
                'anneal_cap': anneal_cap,
                'batch_size': batch_size,
                'ndcg@10': ndcg,
                'training_time': training_time,
                'run_id': run_id
            }
            all_results['MultiVAE'].append(result)

            print(f"NDCG@10: {ndcg:.4f}, Time: {training_time:.2f}s")

            if ndcg > best_multivae_ndcg:
                best_multivae_ndcg = ndcg
                best_multivae_params = result

            # Clear GPU memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        best_results['MultiVAE'] = best_multivae_params

        # Save MultiVAE results immediately
        save_model_results('MultiVAE', all_results['MultiVAE'], best_multivae_params, run_id, dataset)

    # Print final summary
    print_final_summary(best_results, run_id)

    # Save combined summary
    save_combined_summary(all_results, best_results, run_id, dataset, args)


def save_model_results(model_name, model_results, best_params, run_id, dataset):
    """Save results for a specific model immediately after completion"""
    filename = f'hyperparameter_search/results_{model_name}_{run_id}.txt'

    with open(filename, 'w') as f:
        f.write(f"Hyperparameter Search Results - {model_name}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Run ID: {run_id}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset.num_users} users, {dataset.num_items} items\n")
        f.write(f"Train interactions: {len(dataset.train_df)}\n")
        f.write(f"Validation interactions: {len(dataset.val_df)}\n\n")

        # Performance summary
        total_time = sum(r['training_time'] for r in model_results)
        f.write(f"Total {model_name} Search Time: {total_time:.2f} seconds\n")
        f.write(f"Average Time per Experiment: {total_time / len(model_results):.2f} seconds\n")
        f.write(f"Number of Experiments: {len(model_results)}\n\n")

        # Best result
        f.write(f"BEST {model_name} RESULT:\n")
        f.write("-" * 25 + "\n")
        f.write(f"NDCG@10: {best_params['ndcg@10']:.4f}\n")
        f.write("Parameters:\n")
        for key, value in best_params.items():
            if key not in ['model', 'ndcg@10', 'training_time', 'run_id']:
                f.write(f"  {key}: {value}\n")
        f.write(f"Training Time: {best_params['training_time']:.2f} seconds\n\n")

        # All results sorted by NDCG@10
        f.write(f"ALL {model_name} RESULTS (sorted by NDCG@10):\n")
        f.write("-" * 40 + "\n")
        for result in sorted(model_results, key=lambda x: x['ndcg@10'], reverse=True):
            f.write(f"NDCG@10: {result['ndcg@10']:.4f} | ")
            f.write(f"Time: {result['training_time']:.2f}s | ")
            params = {k: v for k, v in result.items()
                      if k not in ['model', 'ndcg@10', 'training_time', 'run_id']}
            f.write(f"Params: {params}\n")

    print(f"✓ {model_name} results saved to '{filename}'")


def print_final_summary(best_results, run_id):
    """Print final summary of all models"""
    print("\n" + "=" * 80)
    print("FINAL HYPERPARAMETER SEARCH SUMMARY")
    print("=" * 80)
    print(f"Run ID: {run_id}")

    if not best_results:
        print("No models were searched.")
        return

    # Find overall best model
    overall_best = max(best_results.items(), key=lambda x: x[1]['ndcg@10'])

    print(f"\n🏆 OVERALL BEST MODEL: {overall_best[0]} (NDCG@10: {overall_best[1]['ndcg@10']:.4f})\n")

    # Print best for each model
    for model_name, best_params in best_results.items():
        print(f"{model_name}: NDCG@10 = {best_params['ndcg@10']:.4f}")
        for key, value in best_params.items():
            if key not in ['model', 'ndcg@10', 'training_time', 'run_id']:
                print(f"  {key}: {value}")
        print(f"  training_time: {best_params['training_time']:.2f}s")
        print()


def save_combined_summary(all_results, best_results, run_id, dataset, args):
    """Save combined summary of all models"""
    filename = f'results_summary_{run_id}.txt'

    with open(filename, 'w') as f:
        f.write("HYPERPARAMETER SEARCH SUMMARY - ALL MODELS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Run ID: {run_id}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Models Searched: {', '.join(args.models)}\n")
        f.write(f"Users: {dataset.num_users}, Items: {dataset.num_items}\n")
        f.write(f"Train interactions: {len(dataset.train_df)}\n")
        f.write(f"Validation interactions: {len(dataset.val_df)}\n\n")

        # Overall statistics
        total_experiments = sum(len(results) for results in all_results.values())
        total_time = sum(sum(r['training_time'] for r in results) for results in all_results.values())

        f.write("SEARCH STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Experiments: {total_experiments}\n")
        f.write(f"Total Search Time: {total_time:.2f} seconds ({total_time / 60:.1f} minutes)\n")
        f.write(f"Average Time per Experiment: {total_time / total_experiments:.2f} seconds\n\n")

    print(f"✓ Finished search")


if __name__ == "__main__":
    hyperparameter_search()

