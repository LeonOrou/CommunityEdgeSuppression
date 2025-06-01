import torch
from torch.utils.data import Dataset
import numpy as np
import os
from LightGCN_PyTorch.code.dataloader import BasicDataset
from os.path import join
import scipy.sparse as sp
from LightGCN_PyTorch.code.world import cprint
from scipy.sparse import csr_matrix
from LightGCN_PyTorch.code import world
from time import time


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
        # TODO: reindex also genre label item id's
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


class Movielens(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    Movielens 100k dataset
    """

    def __init__(self, dataset_name='ml-100k'):
        # train or test
        cprint("loading [ml-100k] from processed file")
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.dataset_name = dataset_name
        self.path = f'dataset/{self.dataset_name}'

        # Load the processed file
        processed_data = np.load(f'{self.path}/{dataset_name}_processed.npy')
        processed_data = np.array(processed_data, dtype=np.int64)
        cprint(f"Loaded data shape: {processed_data.shape}")

        # Extract user, item, and rating information
        users = processed_data[:, 0]
        items = processed_data[:, 1]
        ratings = processed_data[:, 2]

        # Get the number of users and items
        self.n_user = np.max(users) + 1  # IDs start at 0
        self.m_item = np.max(items) + 1

        # Split the data into train and test sets
        # already shuffled in preprocessing
        indices = np.arange(len(users))

        # keep in mind cross validation
        # test is last 20% of the data
        # valid set is switching 16% of the train data
        test_size_from = int(len(users) * 0.8)
        test_indices = indices[test_size_from:]
        train_indices = indices[:test_size_from]

        self.interaction = processed_data
        self.train_interaction = processed_data[train_indices]
        self.test_interaction = processed_data[test_indices]

        # Create train and test datasets
        self.trainUser = users[train_indices]
        self.trainItem = items[train_indices]
        self.trainRating = ratings[train_indices]

        self.testUser = users[test_indices]
        self.testItem = items[test_indices]
        self.testRating = ratings[test_indices]

        self.trainUniqueUsers = np.unique(self.trainUser)
        self.testUniqueUsers = np.unique(self.testUser)

        self.traindataSize = len(self.trainUser)
        self.testDataSize = len(self.testUser)

        # print(f"Movielens Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        print(f"Number of users: {self.n_users}, Number of items: {self.m_items}")
        print(f"Training interactions: {self.traindataSize}, Testing interactions: {self.testDataSize}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)))

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.Graph = None
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        negItems = []
        allItems = set(range(self.m_items))
        for user in users:
            posItems = set(self.getUserPosItems([user])[0])
            negItems.append(np.array(list(allItems - posItems)))
        return negItems

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(join(self.path, 's_pre_adj_mat.npz'))
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(join(self.path, 's_pre_adj_mat.npz'), norm_adj)

            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        return len(self.trainUniqueUsers)

