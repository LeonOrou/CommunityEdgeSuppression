import torch
import logging


class Config:
    """Central configuration class to store all parameters."""

    def __init__(self):
        self.latent_dimension = None
        self.model_name = None
        self.dataset_name = None
        self.users_top_percent = None
        self.items_top_percent = None
        self.users_dec_perc_suppr = None
        self.items_dec_perc_suppr = None
        self.community_suppression = None
        self.suppress_power_nodes_first = None
        self.use_suppression = None
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
        self.patience = 10
        self.gamma = 0.5
        self.min_lr = 1e-5
        self.reproducibility = True
        self.learning_rate = 5e-4
        self.nr_items = None
        self.training_time = None

        self.setup_device()

    def update_from_args(self, args):
        """Update config from command line arguments."""
        for key, value in vars(args).items():
            value = True if value == "True" else value
            value = False if value == "False" else value
            setattr(self, key, value)

    def setup_model_config(self):
        """Setup model-specific configurations."""
        if self.model_name == 'LightGCN':
            self.learning_rate = 5e-3
            self.n_layers = 4
            self.embedding_dim = 128
            self.num_folds = 5
            self.reg = 1e-4
            self.patience = 10

            if self.dataset_name == 'ml-100k':
                self.batch_size = 1024
                self.epochs = 150
                self.n_layers = 4
                self.embedding_dim = 128
            elif self.dataset_name == 'ml-1m':
                self.batch_size = 2048
                self.epochs = 50
                self.n_layers = 4
                self.embedding_dim = 128
            elif self.dataset_name == 'lastfm':
                self.batch_size = 1024
                self.epochs = 150
                self.n_layers = 4
                self.embedding_dim = 128

        elif self.model_name == 'ItemKNN':
            if self.dataset_name == 'ml-100k':
                self.item_knn_topk = 200
                self.shrink = 10
            elif self.dataset_name == 'ml-1m':
                self.item_knn_topk = 50
                self.shrink = 10
            elif self.dataset_name == 'lastfm':
                self.item_knn_topk = 200
                self.shrink = 1
            else:
                self.item_knn_topk = 125
                self.shrink = 30
            self.feature_weighting = 'tf-idf'

        elif self.model_name == 'MultiVAE':
            self.learning_rate = 5e-3
            self.epochs = 200
            self.batch_size = 2048
            self.patience = 10
            if self.dataset_name == 'ml-100k':
                self.hidden_dimension = 800
                self.latent_dimension = 200
                self.anneal_cap = 0.4
            elif self.dataset_name == 'ml-1m':
                self.hidden_dimension = 800
                self.latent_dimension = 200
                self.anneal_cap = 0.3
            elif self.dataset_name == 'lastfm':
                # still check
                self.hidden_dimension = 600
                self.latent_dimension = 200
                self.anneal_cap = 0.4
            self.dropout_prob = 0.5
            self.total_anneal_steps = 100
            # self.weight_decay = 1e-2

    def setup_device(self, try_gpu=True):
        """Setup computation device."""
        if try_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        return self.device





