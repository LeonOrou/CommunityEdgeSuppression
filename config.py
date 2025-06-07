import torch
import logging


class Config:
    """Central configuration class to store all parameters."""

    def __init__(self):
        self.model_name = None
        self.dataset_name = None
        self.users_top_percent = None
        self.items_top_percent = None
        self.users_dec_perc_drop = None
        self.items_dec_perc_drop = None
        self.community_suppression = None
        self.drop_only_power_nodes = None
        self.use_dropout = None
        self.evaluate_top_k = [10, 20, 50, 100]

        # Will be set during initialization
        self.device = None
        self.user_com_labels = None
        self.item_com_labels = None
        self.power_users_ids = None
        self.power_items_ids = None
        self.user_community_connectivity_matrix = None
        self.item_community_connectivity_matrix = None
        self.user_community_connectivity_matrix_distribution = None
        self.item_community_connectivity_matrix_distribution = None
        self.biased_user_edges_mask = None
        self.biased_item_edges_mask = None
        self.train_dataset_len = 0

        # training parameters; lr-scheduler, optimizer, etc.
        self.patience = 2
        self.gamma = 0.5
        self.min_lr = 1e-5
        self.reproducibility = True
        self.learning_rate = 1e-4
        self.nr_items = None

        self.setup_device()

    def update_from_args(self, args):
        """Update config from command line arguments."""
        for key, value in vars(args).items():
            setattr(self, key, value)

    def setup_model_config(self):
        """Setup model-specific configurations."""
        if self.model_name == 'LightGCN':
            self.batch_size = 1024
            self.learning_rate = 0.005
            self.epochs = 100  # because it's different for each model
            self.n_layers = 3
            self.embedding_dim = 64
            self.A_split = False
            self.keep_prob = 0.0  # dropout rate
            self.dropout = False
            self.num_folds = 5
            self.reg = 1e-4
            self.weight_decay = 1
        elif self.model_name == 'ItemKNN':
            self.epochs = 1
            self.item_knn_topk = 250
            self.shrink = 10
            self.feature_weighting = 'bm25'
        elif self.model_name == 'MultiVAE':
            self.epochs = 200
            self.batch_size = 4096  # For consistency
            self.hidden_dimension = 800
            self.latent_dimension = 200
            self.q_dims = [self.hidden_dimension, self.latent_dimension]
            self.p_dims = [self.latent_dimension, self.hidden_dimension, self.nr_items]
            self.drop = 0.7
            self.anneal_cap = 0.3
            self.total_anneal_steps = 200000

    def setup_device(self, try_gpu=True):
        """Setup computation device."""
        if try_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        return self.device





