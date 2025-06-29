import torch
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder


def get_dataset_tensor(config):
    """Get dataset tensor using config object."""
    if not os.path.exists(f'dataset/{config.dataset_name}/{config.dataset_name}_processed.npy'):
        min_degree = 10 if 'ml' in config.dataset_name else 5  # 10 for lfm?
        min_rating = 4 if 'ml' in config.dataset_name else 5  # in lfm rating is number of listening event
        if os.path.exists(f'dataset/{config.dataset_name}/{config.dataset_name}.inter'):
            interaction = np.loadtxt(f'dataset/{config.dataset_name}/{config.dataset_name}.inter', delimiter='\t', skiprows=1)
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
        user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users, start=0)}
        # Create mappings for item IDs
        unique_items = np.unique(interaction[:, 1])
        item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items, start=0)}
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

    user_degrees = np.bincount(interaction[:, 0].astype(int))
    item_degrees = np.bincount(interaction[:, 1].astype(int))
    config.user_degrees = user_degrees
    config.item_degrees = item_degrees

    return torch.tensor(interaction, dtype=torch.int64, device=config.device)


def get_pos_neg_items_from_dataset(dataset):
    """
    Get positive and negative items for each user from the dataset.
    :param dataset:
    :return: np.array: shape=(nr_users, (user_ids, positive items, negative items))
    """
    users = np.arange(dataset.n_user, dtype=np.int64)
    allPos = dataset.allPos
    S = []
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    return np.array(S)


class RecommendationDataset:
    def __init__(self, name='ml-100k', data_path=None, min_interactions=5,
                 test_ratio=0.2, val_ratio=0.16, random_state=42):
        """
        Unified dataset class for recommendation systems

        Args:
            name: Dataset name ('ml-100k', 'ml-20m', 'lfm')
            data_path: Path to dataset files
            min_interactions: Minimum interactions per user/item
            test_ratio: Ratio of test set
            val_ratio: Ratio of validation set
            random_state: Random seed for reproducibility
        """
        self.name = name
        self.data_path = data_path if data_path else f'dataset/{name}/'
        self.min_interactions = min_interactions
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Core attributes (populated after loading)
        self.raw_df = None  # Original ratings dataframe
        self.complete_df = None  # Filtered and encoded dataframe
        self.train_df = None
        self.val_df = None
        self.train_mask = None
        self.fold_masks = None  # Masks for cross-validation folds
        self.n_folds = 5  # Default number of folds for cross-validation

        # Encoding and metadata
        self.num_users = None
        self.num_items = None
        self.user_encoder = None
        self.item_encoder = None
        self.user_degrees = None
        self.item_degrees = None
        self.item_popularities = None

        # Graph structures (for graph-based models)
        self.train_edge_index = None
        self.train_edge_weight = None
        self.complete_edge_index = None
        self.complete_edge_weight = None
        self.current_edge_weight = None  # used for community suppression

        # Interaction mappings
        self.user_positive_items = defaultdict(set)  # {user_id: set(item_ids)}
        self.item_positive_users = defaultdict(set)  # {item_id: set(user_ids)}

        self.ensure_true_negatives = True  # Ensure negative sampling does not include positive items
        self.user_valid_negative_pools = {}
        self.valid_train_indices = None
        self.neg_cumsum = None
        self.neg_sampling_probs = None

        # Statistics
        self.stats = {}

    def load_data(self):
        """Load raw data based on dataset type"""
        if self.name.startswith('ml-'):
            if not os.path.exists(f'dataset/{self.name}/saved'):
                os.makedirs(f'dataset/{self.name}/saved')
            self._load_movielens()
        elif self.name == 'lastfm':
            if not os.path.exists(f'dataset/{self.name}/saved'):
                os.makedirs(f'dataset/{self.name}/saved')
            self._load_lfm()
        else:
            raise ValueError(f"Unknown dataset: {self.name}")

        self.raw_df = self.raw_df.drop_duplicates(subset=['user_id', 'item_id']).reset_index(drop=True)
        # we shuffle later with indices
        self.raw_df = self.raw_df.sample(frac=1, random_state=42).reset_index(drop=True)

        self._compute_statistics()
        return self

    def _load_movielens(self):
        if self.name == 'ml-100k':
            ratings_file = os.path.join(f'{self.data_path}', 'u.data')
            self.raw_df = pd.read_csv(ratings_file, sep='\t',
                                      names=['user_id', 'item_id', 'rating'],
                                      usecols=[0, 1, 2], header=None)
        elif self.name == 'ml-1m':
            ratings_file = os.path.join(self.data_path, 'ratings.csv')
            self.raw_df = pd.read_csv(ratings_file, sep='::',
                                      names=['user_id', 'item_id', 'rating'],
                                      usecols=[0, 1, 2], header=0)
        elif self.name == 'ml-20m':
            ratings_file = os.path.join(self.data_path, 'ratings.csv')
            self.raw_df = pd.read_csv(ratings_file, usecols=['user_id', 'item_id', 'rating'],
                                      names=['user_id', 'item_id', 'rating'], header=0)

    def _load_lfm(self):
        if self.name == 'lastfm':
            ratings_file = os.path.join(self.data_path, 'user_artists.csv')
            self.raw_df = pd.read_csv(ratings_file, sep='\t',
                                      names=['user_id', 'item_id', 'rating'],  # its count not rating but for common naming
                                      usecols=[0, 1, 2], header=0)
        self.raw_df = self.raw_df.sample(frac=1, random_state=42).reset_index(drop=True)

    def calculate_item_popularities(self):
        """Calculate item popularities based on interaction counts"""
        if self.complete_df is not None:
            item_counts = self.complete_df['item_encoded'].value_counts()
            self.item_popularities = item_counts
        else:
            raise ValueError("complete_df is not set. Please prepare the data first.")

    def prepare_data_with_consistent_encoding(self):
        """
        Prepare data ensuring ALL users and items are encoded consistently
        across train/validation/test splits
        """
        ratings_df = self.raw_df.copy()
        min_interactions = 5 if self.name.startswith('ml-') else 5  # for lastfm, minimum interactions per user/item
        min_rating = 4 if self.name.startswith('ml-') else 2  # for lastfm, rating is number of listening events

        user_counts = ratings_df['user_id'].value_counts()
        item_counts = ratings_df['item_id'].value_counts()

        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index

        filtered_df = ratings_df[
            (ratings_df['user_id'].isin(valid_users)) &
            (ratings_df['item_id'].isin(valid_items))
            ].copy()

        filtered_df = filtered_df[filtered_df['rating'] >= min_rating].reset_index(drop=True)

        filtered_df['rating'] = 1  # Treat all valid ratings as positive interactions

        # This ensures consistent encoding across all splits
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()

        filtered_df['user_encoded'] = user_encoder.fit_transform(filtered_df['user_id'])
        filtered_df['item_encoded'] = item_encoder.fit_transform(filtered_df['item_id'])

        # save encoders for genre label encoding
        if not os.path.exists(f'dataset/{self.name}/encoders'):
            os.makedirs(f'dataset/{self.name}/encoders')
        pd.to_pickle(user_encoder, f'dataset/{self.name}/encoders/user_encoder.pkl')
        pd.to_pickle(item_encoder, f'dataset/{self.name}/encoders/item_encoder.pkl')
        pd.to_pickle(filtered_df, f'dataset/{self.name}/saved/filtered_df.pkl')

        num_users = len(user_encoder.classes_)
        num_items = len(item_encoder.classes_)

        self.complete_df = filtered_df
        self.num_users = num_users
        self.num_items = num_items

    def split_interactions_by_user(self, n_folds=5):
        """
        Create masks for n_folds subsets of the complete_df, where each subset
        has at least one edge from each user.

        Parameters:
        -----------
        n_folds : int
            Number of folds to create (default: 5)
        """
        # Initialize random state for reproducibility
        rng = np.random.RandomState(42)

        # Get the dataframe
        df = self.complete_df.copy()
        n_total = len(df)

        # Get unique users
        unique_users = df['user_encoded'].unique()
        n_users = len(unique_users)

        # Initialize fold masks
        self.fold_masks = np.zeros((n_folds, n_total), dtype=bool)

        # Track which interactions have been assigned
        assigned = np.zeros(n_total, dtype=bool)

        # First, ensure each fold has at least one interaction per user
        for fold_idx in range(n_folds):
            # For each user, assign one random interaction to this fold
            for user in unique_users:
                user_mask = (df['user_encoded'].values == user) & (~assigned)
                user_indices = np.where(user_mask)[0]

                if len(user_indices) > 0:
                    # Select a random interaction for this user
                    selected_idx = rng.choice(user_indices)
                    self.fold_masks[fold_idx, selected_idx] = True
                    assigned[selected_idx] = True

        # Distribute remaining interactions randomly across folds
        remaining_indices = np.where(~assigned)[0]
        if len(remaining_indices) > 0:
            # Shuffle remaining indices
            rng.shuffle(remaining_indices)

            # Calculate how many to assign to each fold
            base_size = len(remaining_indices) // n_folds
            remainder = len(remaining_indices) % n_folds

            start_idx = 0
            for fold_idx in range(n_folds):
                # Determine size for this fold
                fold_size = base_size + (1 if fold_idx < remainder else 0)

                # Assign indices to this fold
                end_idx = start_idx + fold_size
                fold_indices = remaining_indices[start_idx:end_idx]
                self.fold_masks[fold_idx, fold_indices] = True

                start_idx = end_idx

        # Verify that each fold has all users
        # for fold_idx in range(n_folds):
        #     fold_df = df[self.fold_masks[fold_idx]]
        #     fold_users = fold_df['user_encoded'].unique()
        #
        #     assert len(fold_users) == n_users, f"Fold {fold_idx} missing users"

        # print(f"Created {n_folds} fold masks")
        # for fold_idx in range(n_folds):
        #     print(f"Fold {fold_idx}: {self.fold_masks[fold_idx].sum()} interactions")

    def get_fold_i(self, i, n_folds=5):
        """
        Generate train and validation dataframes for fold i.
        Uses pre-computed masks from split_interactions_by_user_item.

        Parameters:
        -----------
        i : int
            Index of the fold (0 to n_folds-1)
        n_folds : int
            Total number of folds (default: 5)

        Returns:
        --------
        None (sets self.train_df and self.val_df)
        """
        if i < 0 or i >= n_folds:
            raise ValueError(f"Fold index must be between 0 and {n_folds - 1}, got {i}")

        # Check if fold masks exist, if not create them
        if not hasattr(self, 'fold_masks'):
            print(f"Fold masks not found. Creating {n_folds} fold masks...")
            self.split_interactions_by_user_item(n_folds=n_folds)

        # Get the i-th mask for validation
        val_mask = self.fold_masks[i]

        # Combine other folds for training
        train_mask = np.zeros(len(self.complete_df), dtype=bool)
        for fold_idx in range(n_folds):
            if fold_idx != i:
                train_mask |= self.fold_masks[fold_idx]

        # Create train and validation dataframes
        self.val_df = self.complete_df[val_mask].copy()
        self.train_df = self.complete_df[train_mask].copy()

        # Store masks for potential later use
        self.train_mask = train_mask
        self.val_mask = val_mask

        # print(f"Fold {i}: Train size = {len(self.train_df)}, Val size = {len(self.val_df)}")

        # # Verify all users and items are in both sets
        # train_users = self.train_df['user_encoded'].nunique()
        # train_items = self.train_df['item_encoded'].nunique()
        # val_users = self.val_df['user_encoded'].nunique()
        # val_items = self.val_df['item_encoded'].nunique()
        #
        # total_users = self.complete_df['user_encoded'].nunique()
        # total_items = self.complete_df['item_encoded'].nunique()
        #
        # print(f"Train: {train_users}/{total_users} users, {train_items}/{total_items} items")
        # print(f"Val: {val_users}/{total_users} users, {val_items}/{total_items} items")

    def prepare_data(self):
        """Filter, encode, and split data"""

        # load if processed file is already available
        if not os.path.exists(f'dataset/{self.name}/saved/filtered_df.pkl'):
            self.load_data()
            self.prepare_data_with_consistent_encoding()
            self.raw_df = None  # Clear raw_df to save memory
        else:
            self.complete_df = pd.read_pickle(f'dataset/{self.name}/saved/filtered_df.pkl')
            self.num_users = len(pd.unique(self.complete_df['user_encoded']))
            self.num_items = len(pd.unique(self.complete_df['item_encoded']))

        self.split_interactions_by_user()
        self.calculate_item_popularities()


    def _create_graph_structures(self, rating_weights=None):
        """Create graph structures for graph-based models"""
        self.complete_edge_index, self.complete_edge_weight = self.create_bipartite_graph(df=self.complete_df, edge_weights=rating_weights)

    def _build_interaction_mappings(self):
        """Build user-item interaction mappings"""
        for df in [self.train_df, self.val_df, self.test_df]:
            if df is not None and len(df) > 0:
                for _, row in df.iterrows():
                    user_id = row['user_encoded']
                    item_id = row['item_encoded']
                    self.user_positive_items[user_id].add(item_id)
                    self.item_positive_users[item_id].add(user_id)

    def get_train_interactions(self, user_id=None):
        """Get training interactions for a specific user or all users"""
        if user_id is not None:
            user_items = self.train_df[self.train_df['user_encoded'] == user_id]
            return set(user_items['item_encoded'].values)
        return self.train_df

    def get_user_positive_items(self, user_id, split='all'):
        """Get positive items for a user"""
        if split == 'train':
            return set(self.train_df[self.train_df['user_encoded'] == user_id]['item_encoded'].values)
        elif split == 'all':
            return self.user_positive_items.get(user_id, set())

    def sample_negative_items(self, user_ids):
        """Sample negative items for given users"""
        neg_items = torch.zeros(len(user_ids), dtype=torch.int64, device=self.device)
        for i, user_id in enumerate(user_ids):
            user_pos_items = self.get_user_positive_items(user_id, split='all')

            neg_item = torch.randint(0, self.num_items, (1,))
            attempts = 0
            while neg_item in user_pos_items and attempts < 100:
                neg_item = torch.randint(0, self.num_items, (1,))
                attempts += 1
            neg_items[i] = neg_item

        return neg_items

    def build_masked_negative_pools(self):
        """Pre-compute negative pools considering train_mask"""
        # Count item frequencies in positive interactions
        item_counts = torch.zeros(self.num_items, device=self.device)
        for user_id in range(self.num_users):
            pos_items = self.get_user_positive_items(user_id, split='all')
            for item in pos_items:
                item_counts[item] += 1

        # Create sampling probabilities (inverse popularity)
        neg_probs = 1.0 / (item_counts + 1.0)  # Add 1 to avoid division by zero
        neg_probs = neg_probs / neg_probs.sum()

        # Store cumulative distribution
        self.neg_sampling_probs = neg_probs
        self.neg_cumsum = torch.cumsum(neg_probs, dim=0)

    def sample_negative_items_pool(self, user_ids):
        """Sample negative items for given users"""
        batch_size = len(user_ids)

        # Sample from distribution
        uniform_samples = torch.rand(batch_size, device=self.device)
        neg_items = torch.searchsorted(self.neg_cumsum, uniform_samples)

        # Verify and resample if needed (vectorized)
        for i, user_id in enumerate(user_ids):
            user_pos_items = set(self.get_user_positive_items(user_id, split='all'))

            # Resample if we hit a positive item
            attempts = 0
            while neg_items[i].item() in user_pos_items and attempts < 10:
                uniform_sample = torch.rand(1, device=self.device)
                neg_items[i] = torch.searchsorted(self.neg_cumsum, uniform_sample)
                attempts += 1

        return neg_items

    def sample_negative_items_batch(self, user_ids, num_negatives=1):
        """
        Ultra-fast negative sampling that allows false negatives
        with very low probability. Best for large-scale training.
        """
        batch_size = len(user_ids)

        # if not hasattr(self, 'valid_train_indices'):
        #     valid_train_indices = torch.where(self.train_mask)[0]
        #     num_valid_items = len(valid_train_indices)

        neg_items = torch.randint(0, self.num_items,
                                  (batch_size, num_negatives),
                                  device=self.device)

        # If you need to ensure no positives (slower but still fast)
        if self.ensure_true_negatives:
            for i, user_id in enumerate(user_ids):
                user_pos_items = set(self.get_user_positive_items(user_id, split='all'))
                for j in range(num_negatives):
                    if neg_items[i, j].item() in user_pos_items:
                        # Sample from negative pool
                        all_items = np.arange(self.num_items)
                        mask = np.ones(self.num_items, dtype=bool)
                        mask[list(user_pos_items)] = False
                        neg_candidates = all_items[mask]
                        if len(neg_candidates) > 0:
                            neg_items[i, j] = torch.tensor(
                                np.random.choice(neg_candidates),
                                device=self.device
                            )

        return neg_items.squeeze() if num_negatives == 1 else neg_items

    def get_dataloader(self, split='train', batch_size=512, shuffle=True,
                       num_negatives=1, model_type='lightgcn'):
        """Get dataloader for specific split and model type"""
        if model_type in ['lightgcn', 'itemknn']:
            return self._get_pairwise_dataloader(split, batch_size, shuffle, num_negatives)
        elif model_type == 'multivae':
            return self._get_pointwise_dataloader(split, batch_size, shuffle)

    def _get_pairwise_dataloader(self, split, batch_size, shuffle, num_negatives):
        """Get dataloader for pairwise models (LightGCN, ItemKNN)"""
        df = getattr(self, f'{split}_df')

        if shuffle:
            df = df.sample(frac=1)

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]

            users = batch['user_encoded'].values
            pos_items = batch['item_encoded'].values
            neg_items = self.sample_negative_items(users, num_negatives)

            yield {
                'users': torch.tensor(users, dtype=torch.long),
                'pos_items': torch.tensor(pos_items, dtype=torch.long),
                'neg_items': torch.tensor(neg_items, dtype=torch.long),
                'batch_size': len(batch)
            }

    def _get_pointwise_dataloader(self, split, batch_size, shuffle):
        """Get dataloader for pointwise models (MultiVAE)"""
        # Create user-item interaction matrix for VAE
        df = getattr(self, f'{split}_df')

        # Create sparse matrix representation
        from scipy.sparse import csr_matrix

        rows = df['user_encoded'].values
        cols = df['item_encoded'].values
        data = np.ones(len(df))  # Binary for MultiVAE

        interaction_matrix = csr_matrix((data, (rows, cols)),
                                        shape=(self.num_users, self.num_items))

        user_ids = np.arange(self.num_users)
        if shuffle:
            np.random.shuffle(user_ids)

        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i + batch_size]
            batch_matrix = interaction_matrix[batch_users].toarray()

            yield {
                'users': torch.tensor(batch_users, dtype=torch.long),
                'interactions': torch.tensor(batch_matrix, dtype=torch.float32),
                'batch_size': len(batch_users)
            }

    def _compute_statistics(self):
        """Compute dataset statistics"""
        if self.complete_df is not None:
            self.stats = {
                'num_users': self.num_users,
                'num_items': self.num_items,
                'num_interactions': len(self.complete_df),
                'density': len(self.complete_df) / (self.num_users * self.num_items),
                'avg_interactions_per_user': len(self.complete_df) / self.num_users,
                'avg_interactions_per_item': len(self.complete_df) / self.num_items,
            }

    def to_device(self, device):
        """Move graph data to device"""
        if self.train_edge_index is not None:
            self.train_edge_index = self.train_edge_index.to(device)
            self.train_edge_weight = self.train_edge_weight.to(device)
        if self.complete_edge_index is not None:
            self.complete_edge_index = self.complete_edge_index.to(device)
            self.complete_edge_weight = self.complete_edge_weight.to(device)
        return self

    def create_bipartite_graph(self, df, edge_weights=None, device='cpu'):
        """Create bipartite graph with edge weights based on ratings"""
        # df = self.complete_df
        users = df['user_encoded'].values
        items = df['item_encoded'].values + self.num_users  # Offset items by num_users
        if edge_weights is None:
            edge_weights = df['rating'].values

        # Create bidirectional edges (user->item and item->user)
        edge_index = torch.stack([
            torch.tensor(np.concatenate([users, items]), dtype=torch.long),
            torch.tensor(np.concatenate([items, users]), dtype=torch.long)
        ], dim=0)

        edge_weight = torch.tensor(
            np.concatenate([edge_weights, edge_weights]),
            dtype=torch.float
        )

        return edge_index.to(device), edge_weight.to(device)

    def get_node_degrees(self):
        """Get degrees of users and items in the graph"""
        user_degrees = np.bincount(self.complete_df['user_encoded'])
        item_degrees = np.bincount(self.complete_df['item_encoded'])
        self.user_degrees = user_degrees
        self.item_degrees = item_degrees
        return user_degrees, item_degrees


def sample_negative_items(user_ids, pos_item_ids, num_items, user_positive_items, device):
    batch_size = len(user_ids)
    neg_items = np.zeros(batch_size, dtype=np.int32)  # Use int32

    # Process on CPU for efficiency with sparse data
    user_ids_cpu = user_ids.cpu().numpy()

    for i in range(batch_size):
        user_id = user_ids_cpu[i]
        user_pos_set = user_positive_items.get(user_id, set())

        # Simple and fast negative sampling
        neg_item = np.random.randint(0, num_items)
        attempts = 0
        while neg_item in user_pos_set and attempts < 100:
            neg_item = np.random.randint(0, num_items)
            attempts += 1

        neg_items[i] = neg_item

    # Single transfer to GPU with int32
    return torch.tensor(neg_items, dtype=torch.int32, device=device)


def prepare_adj_tensor(dataset):
    """Prepare adjacency tensor from dataset in the format (user_id, item_id, rating)"""
    df = dataset.complete_df
    adj_tens = torch.tensor(
        np.column_stack([
            df['user_encoded'].values,
            df['item_encoded'].values,
            df['rating'].values
        ]),
        dtype=torch.int64,
        device=dataset.device
    )
    return adj_tens


def prepare_training_data(train_df, device):
    """Pre-convert training data to GPU tensors with memory-efficient dtypes"""
    # Use int32 for user/item indices - sufficient for millions of users/items
    all_users = torch.tensor(train_df['user_encoded'].values, dtype=torch.int32, device=device)
    all_items = torch.tensor(train_df['item_encoded'].values, dtype=torch.int32, device=device)

    # Create indices for shuffling (needs long for arange)
    indices = torch.arange(len(train_df), dtype=torch.long, device=device)

    return all_users, all_items, indices


