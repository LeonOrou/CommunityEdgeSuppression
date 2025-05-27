from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import KFold
from LightGCN_PyTorch.code.utils import minibatch
from evaluation import evaluate_model
from evaluation import precalculate_average_popularity
from utils_functions import community_edge_dropout
import wandb
import numpy as np
from dataset import get_pos_neg_items_from_dataset
import os.path as osp
import torch
from tqdm import tqdm
from torch_geometric.datasets import MovieLens100K
from torch_geometric.nn import LightGCN
from torch_geometric.utils import degree


def bpr_loss(self, users, pos, neg):
    (users_emb, pos_emb, neg_emb,
    userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
    reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                     posEmb0.norm(2).pow(2)  +
                     negEmb0.norm(2).pow(2))/float(len(users))
    pos_scores = torch.mul(users_emb, pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(users_emb, neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)

    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

    return loss, reg_loss


def get_dataset_loader(dataset, batch_size, device):
    """
    Creates a loader for LightGCN with device awareness

    Args:
        dataset: LightGCN dataset with user-item triplets
        batch_size: Size of batches
        device: Device to place tensors on (from config.device)

    Returns:
        Batched data loader for training
    """
    users = torch.tensor(dataset[:, 0], dtype=torch.long).to(device)
    pos_items = torch.tensor(dataset[:, 1], dtype=torch.long).to(device)
    neg_items = torch.tensor(dataset[:, 2], dtype=torch.long).to(device)

    data_loader = minibatch(users, pos_items, neg_items, batch_size=batch_size)
    return data_loader


def train_and_evaluate(config, model, dataset):
    """Train and evaluate model using config for all parameters"""

    # Create a learning rate scheduler that reduces LR on plateau


    best_valid_score = -float('inf')

    kf = KFold(n_splits=5, shuffle=True)
    results = {}
    # calculate average popularity dict
    # S = get_pos_neg_items_from_dataset(dataset)
    # users = torch.tensor(S[:, 0], dtype=torch.int64)
    # items = torch.tensor(S[:, 1], dtype=torch.int64)
    # neg_items = torch.tensor(S[:, 2], dtype=torch.int64)
    #
    # pos_neg_train_dataset = torch.stack((users, items, neg_items), dim=1)
    # shuffle_indexes = torch.randperm(users.shape[0])
    # pos_neg_train_dataset = pos_neg_train_dataset[shuffle_indexes]

    model = LightGCN(
        num_nodes=dataset.n_users + dataset.m_items,
        embedding_dim=config.latent_dim_rec,
        num_layers=config.lightGCN_n_layers,
    ).to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # max ndcg
        factor=config.gamma,
        patience=config.patience,
        min_lr=config.min_lr,
    )
    dataset = MovieLens100K(f'/Datasets/{config.dataset_name}')
    data = dataset[0]
    num_users, num_books = data['user'].num_nodes, data['item'].num_nodes
    data = data.to_homogeneous().to(config.device)

    # Use all message passing edges as training labels:
    batch_size = 512
    mask = data.edge_index[0] < data.edge_index[1]
    train_edge_label_index = data.edge_index[:, mask]

    for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")
        print("-------")
        # TODO: make custom dataloaders for each model
        # TODO: make custom training for each model
        # valid_loader = get_dataset_loader(
        #     dataset=pos_neg_train_dataset[valid_idx],
        #     batch_size=config.batch_size,
        #     device=config.device)

        for epoch in range(config.epochs):
            model.train()
            train_keepers_mask_i = community_edge_dropout(
                adj_tens=dataset.train_interaction,
                config=config)

            interactions = dataset.train_interaction[train_keepers_mask_i]
            user_ids = interactions[:, 0]
            item_ids = interactions[:, 1] + dataset.n_user  # Offset item indices to make unique node ids
            edge_index = torch.stack([user_ids, item_ids], dim=0)  # shape [2, n_edges]
            mask = user_ids < item_ids
            train_edge_label_index = edge_index[:, mask]

            train_loader = torch.utils.data.DataLoader(
                range(train_edge_label_index.size(1)),
                shuffle=True,
                batch_size=config.batch_size,
            )

            # users_train_epoch_i = dataset.trainUser[train_keepers_mask_i]
            # positive_train_items = dataset.getUserPosItems(users_train_epoch_i)
            # negative_train_items = dataset.getUserNegItems(users_train_epoch_i)
            # pos_neg_train_dataset = torch.stack((users_train_epoch_i, positive_train_items, negative_train_items))

            # we make the loader after the dropout to not edit the loader
            # train_loader = get_dataset_loader(pos_neg_train_dataset[train_keepers_mask_i], config.batch_size, config.device)

            epoch_loss = 0.0
            total_loss = 0.0
            total_examples = 0
            for index in tqdm(train_loader):
                pos_edge_label_index = train_edge_label_index[:, index]
                neg_edge_label_index = torch.stack([
                    pos_edge_label_index[0],
                    torch.randint(dataset.n_user, dataset.n_user + dataset.m_items,
                                  (index.numel(),), device=config.device)
                ], dim=0)
                edge_label_index = torch.cat([
                    pos_edge_label_index,
                    neg_edge_label_index,
                ], dim=1)

                optimizer.zero_grad()
                # loss, reg_loss = model.bpr_loss(users, pos_items, neg_items)
                # total_loss = loss + config.weight_decay * reg_loss
                pos_rank, neg_rank = model(torch.arange(torch.sum(train_keepers_mask_i)), edge_label_index).chunk(2)
                loss = model.recommendation_loss(
                    pos_rank,
                    neg_rank,
                    node_id=edge_label_index.unique(),
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss

                total_loss += float(loss) * pos_rank.numel()
                total_examples += pos_rank.numel()

            precision, recall = test(k=20, model=model,
                                     data=dataset.test_interaction,
                                     batch_size,
                                     train_edge_label_index,
                                     num_users)
            print(f'Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}, Precision@20: '
                  f'{precision:.4f}, Recall@20: {recall:.4f}')

            metrics_eval = evaluate_model(model=model, test_loader=valid_loader, device=config.device, item_popularity=avg_item_pop)
            results[fold] = metrics_eval
            # Update learning rate scheduler based on the validation metric
            scheduler.step(metrics_eval[10]['ndcg'])

            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"learning_rate": current_lr})

            if metrics_eval[10]['ndcg'] > best_valid_score:  # eval at ndcg@10
                best_valid_score = metrics_eval['ndcg']
                torch.save(model.state_dict(), f"{config.model_name}_{config.dataset_name}_best.pth")

            wandb.log({"epoch": epoch, "loss": epoch_loss/config.epochs, **metrics_eval})

    results_folds_averages_each_k = {
        k: {metric: sum(fold[k][metric] for fold in results.values()) / len(results) for metric in
            next(iter(results.values()))[k].keys()} for k in next(iter(results.values())).keys()}

    # test the finished model
    test_loader = get_dataset_loader(
        dataset=test_dataset,
        batch_size=config.eval_batch_size,
        device=config.device)

    test_metrics = evaluate_model(model, test_loader, config)
    return best_valid_score, test_metrics


@torch.no_grad()
def test(k: int, model, data, batch_size, train_edge_label_index, num_users):
    emb = model.get_embedding(torch.arange(torch.len(train_edge_label_index)))
    user_emb, book_emb = emb[:num_users], emb[num_users:]

    precision = recall = total_examples = 0
    for start in range(0, num_users, batch_size):
        end = start + batch_size
        logits = user_emb[start:end] @ book_emb.t()

        # Exclude training edges:
        mask = ((train_edge_label_index[0] >= start) &
                (train_edge_label_index[0] < end))
        logits[train_edge_label_index[0, mask] - start,
               train_edge_label_index[1, mask] - num_users] = float('-inf')

        # Computing precision and recall:
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((data.edge_label_index[0] >= start) &
                (data.edge_label_index[0] < end))
        ground_truth[data.edge_label_index[0, mask] - start,
                     data.edge_label_index[1, mask] - num_users] = True
        node_count = degree(data.edge_label_index[0, mask] - start,
                            num_nodes=logits.size(0))

        topk_index = logits.topk(k, dim=-1).indices
        isin_mat = ground_truth.gather(1, topk_index)

        precision += float((isin_mat.sum(dim=-1) / k).sum())
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
        total_examples += int((node_count > 0).sum())

    return precision / total_examples, recall / total_examples
