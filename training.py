from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import KFold

from LightGCN_PyTorch.code.utils import minibatch
from evaluation import evaluate_model
from evaluation import precalculate_average_popularity
from utils_functions import power_node_edge_dropout
from dataset import LightGCNDataset
import wandb


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


def get_train_loader(dataset, batch_size, device):
    """
    Creates a train loader for LightGCN with device awareness

    Args:
        dataset: LightGCN dataset with user-item triplets
        batch_size: Size of batches
        device: Device to place tensors on (from config.device)

    Returns:
        Batched data loader for training
    """
    users = torch.tensor([sample[0] for sample in dataset], dtype=torch.long).to(device)
    pos_items = torch.tensor([sample[1] for sample in dataset], dtype=torch.long).to(device)
    neg_items = torch.tensor([sample[2] for sample in dataset], dtype=torch.long).to(device)

    # Using the minibatch function from utils.py
    train_loader = minibatch(users, pos_items, neg_items, batch_size=batch_size)
    return train_loader


def train_and_evaluate(config, model, train_dataset, test_dataset):
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

    criterion = nn.BCEWithLogitsLoss()
    best_valid_score = -float('inf')

    kf = KFold(n_splits=5, shuffle=True)
    results = {}
    # calculate average popularity dict
    avg_item_pop = precalculate_average_popularity(train_dataset.adj)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_dataset)):
        print(f"Fold {fold + 1}")
        print("-------")

        # TODO: make custom dataloaders for each model
        # TODO: make custom training for each model
        valid_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(valid_idx),
        )

        for epoch in range(config.epochs):
            model.train()
            train_adj_i = power_node_edge_dropout(
                train_dataset.adj,
                power_users_idx=config.power_users_ids,
                power_items_idx=config.power_items_ids,
                biased_user_edges_mask=config.biased_user_edges_mask,
                biased_item_edges_mask=config.biased_item_edges_mask,
                drop_only_power_nodes=config.drop_only_power_nodes,
                community_suppression=config.community_suppression,
                users_dec_perc_drop=config.users_dec_perc_drop,
                items_dec_perc_drop=config.items_dec_perc_drop
            )
            train_dataset_epoch_i = LightGCNDataset(train_adj_i)

            # we make the loader after the dropout to not edit the loader
            train_loader = get_train_loader(train_dataset_epoch_i, config.batch_size, config.device)

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

            wandb.log({"epoch": epoch, "loss": epoch_loss, **metrics_eval})

    results_folds_averages_each_k = {
        k: {metric: sum(fold[k][metric] for fold in results.values()) / len(results) for metric in
            next(iter(results.values()))[k].keys()} for k in next(iter(results.values())).keys()}

    # test the finished model
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_metrics = evaluate_model(model, test_loader, config)
    return best_valid_score, test_metrics