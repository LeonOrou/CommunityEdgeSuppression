import torch
from torch.utils.data import Dataset
import numpy as np
import os
from LightGCN_PyTorch.code.utils import minibatch


class LightGCNDataset(Dataset):
    def __init__(self, interaction_data):
        self.interaction = interaction_data  # Numpy array of [user, item, rating]
        self.n_users = int(interaction_data[:, 0].max()) + 1
        self.n_items = int(interaction_data[:, 1].max()) + 1

        # Build positive item dictionary for each user
        self.user_pos_items = {}
        for user, item, _ in interaction_data:
            if user not in self.user_pos_items:
                self.user_pos_items[user] = []
            self.user_pos_items[user].append(item)

    def __len__(self):
        return len(self.interaction)

    def __getitem__(self, idx):
        user = self.interaction[idx, 0]
        # Sample a positive item
        pos_items = self.user_pos_items[user]
        pos_idx = np.random.randint(0, len(pos_items))
        pos_item = pos_items[pos_idx]

        # Sample a negative item (not in user's positive items)
        while True:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in self.user_pos_items.get(user, []):
                break

        return user, pos_item, neg_item


def get_train_loader(dataset, batch_size):
    users = torch.tensor([sample[0] for sample in dataset], dtype=torch.long).to('cuda')
    pos_items = torch.tensor([sample[1] for sample in dataset], dtype=torch.long).to('cuda')
    neg_items = torch.tensor([sample[2] for sample in dataset], dtype=torch.long).to('cuda')

    # Using the minibatch and shuffle functions from utils.py
    train_loader = minibatch(users, pos_items, neg_items, batch_size=batch_size)
    return train_loader


def get_dataset_tensor(config):
    """Get dataset tensor using config object."""
    if not os.path.exists(f'dataset/{config.dataset_name}/{config.dataset_name}_processed.npy'):
        min_degree = 10 if 'ml' in config.dataset_name else 5  # 10 for lfm?
        min_rating = 4 if 'ml' in config.dataset_name else 5  # in lfm rating is number of listening event
        if os.path.exists(f'dataset/{config.dataset_name}/{config.dataset_name}.inter'):
            interaction = np.loadtxt(f'dataset/{config.dataset_name}/{config.dataset_name}.inter', delimiter=' ', skiprows=1)
        elif os.path.exists(f'dataset/{config.dataset_name}/{config.dataset_name}.data'):
            interaction = np.loadtxt(f'dataset/{config.dataset_name}/{config.dataset_name}.data', delimiter='\t', skiprows=0)
        else:
            raise FileNotFoundError(f"Dataset file not found in dataset/{config.dataset_name}")
        interaction = interaction[:, :3]  # get only user_id, item_id, rating columns
        # if all are 1, we need to binarize the ratings
        interaction = interaction[interaction[:, 2] >= min_rating]  # get only ratings with 4 and above
        interaction[:, 2] = 1  # binarize the ratings
        user_degrees = np.bincount(interaction[:, 0].astype(int))
        item_degrees = np.bincount(interaction[:, 1].astype(int))
        valid_users = np.where(user_degrees >= min_degree)[0]
        valid_items = np.where(item_degrees >= min_degree)[0]
        interaction = interaction[np.isin(interaction[:, 0], valid_users) & np.isin(interaction[:, 1], valid_items)]

        # Create mappings for user IDs
        unique_users = np.unique(interaction[:, 0])
        user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users, start=1)}
        # Create mappings for item IDs
        unique_items = np.unique(interaction[:, 1])
        item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items, start=1)}
        # Create reindexed array
        reindexed = interaction.copy()
        reindexed[:, 0] = np.array([user_id_map[uid] for uid in interaction[:, 0]])
        reindexed[:, 1] = np.array([item_id_map[iid] for iid in interaction[:, 1]])
        interaction = reindexed

        np.random.shuffle(interaction)
        interaction = np.array(interaction, dtype=np.int32)
        np.save(f'dataset/{config.dataset_name}/{config.dataset_name}_processed.npy', interaction)
    else:
        interaction = np.load(f'dataset/{config.dataset_name}/{config.dataset_name}_processed.npy', mmap_mode='r')

    config.train_dataset_len = len(interaction)

    return torch.tensor(interaction, dtype=torch.int64, device=config.device)

