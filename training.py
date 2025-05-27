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
from LightGCN_PyTorch.code.utils import UniformSample_original_python


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
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # Create a learning rate scheduler that reduces LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # max ndcg
        factor=config.gamma,
        patience=config.patience,
        min_lr=config.min_lr,
    )

    best_valid_score = -float('inf')

    kf = KFold(n_splits=5, shuffle=True)
    results = {}
    # calculate average popularity dict
    S = UniformSample_original_python(dataset)
    users = torch.tensor(S[:, 0], dtype=torch.int64)
    items = torch.tensor(S[:, 1], dtype=torch.int64)
    neg_items = torch.tensor(S[:, 2], dtype=torch.int64)
    pos_neg_train_dataset = torch.stack((users, items, neg_items), dim=1)
    train_dataset = get_dataset_loader(pos_neg_train_dataset, config.batch_size, config.device)

    positive_train_items = dataset.getUserPosItems(dataset.trainUser)
    negative_train_items = dataset.getUserNegItems(dataset.trainUser)
    pos_neg_train_dataset = torch.tensor(np.concatenate((dataset.trainUser, positive_train_items, negative_train_items), axis=1), dtype=torch.int64)
    train_dataset = get_dataset_loader(pos_neg_train_dataset, config.batch_size, config.device)

    positive_test_items = dataset.getUserPosItems(dataset.testUser)
    negative_test_items = dataset.getUserNegItems(dataset.testUser)
    pos_neg_test_dataset = torch.tensor(np.concatenate((dataset.testUser, positive_test_items, negative_test_items), axis=1), dtype=torch.int64)
    test_dataset = get_dataset_loader(pos_neg_test_dataset, config.eval_batch_size, config.device)
    avg_item_pop = precalculate_average_popularity(dataset.train_interaction)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")
        print("-------")

        # TODO: make custom dataloaders for each model
        # TODO: make custom training for each model
        valid_loader = get_dataset_loader(
            dataset=pos_neg_train_dataset[valid_idx],
            batch_size=config.batch_size,
            device=config.device)

        for epoch in range(config.epochs):
            model.train()
            train_keepers_mask_i = community_edge_dropout(
                adj_tens=dataset.train_interaction,
                config=config)
            users_train_epoch_i = dataset.trainUser[train_keepers_mask_i]
            positive_train_items = dataset.getUserPosItems(users_train_epoch_i)
            negative_train_items = dataset.getUserNegItems(users_train_epoch_i)
            pos_neg_train_dataset = torch.stack((users_train_epoch_i, positive_train_items, negative_train_items))

            # we make the loader after the dropout to not edit the loader
            train_loader = get_dataset_loader(pos_neg_train_dataset[train_keepers_mask_i], config.batch_size, config.device)

            epoch_loss = 0.0
            for users, pos_items, neg_items in train_loader:
                optimizer.zero_grad()
                loss, reg_loss = model.bpr_loss(users, pos_items, neg_items)
                total_loss = loss + config.weight_decay * reg_loss
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()

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