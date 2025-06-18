import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import degree
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import scipy.sparse as sp


# Custom LightGCN implementation with edge weights
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=12, num_layers=3, reg=1e-5):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings with Xavier uniform
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, edge_index, edge_weight=None):
        # Get initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        # Combine user and item embeddings
        all_emb = torch.cat([user_emb, item_emb], dim=0)

        # Store embeddings from each layer
        embeddings = [all_emb]

        for _ in range(self.num_layers):
            all_emb = self.propagate(edge_index, x=all_emb, edge_weight=edge_weight)
            embeddings.append(all_emb)

        # Average embeddings across all layers
        final_embedding = torch.stack(embeddings, dim=1).mean(dim=1)

        # Split back into user and item embeddings
        user_final = final_embedding[:self.num_users]
        item_final = final_embedding[self.num_users:]

        return user_final, item_final

    def propagate(self, edge_index, x, edge_weight=None):
        # Normalize by degree
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Symmetric normalization
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        if edge_weight is not None:
            norm = norm * edge_weight

        # Message passing
        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row] * norm.view(-1, 1))

        return out

    def predict(self, user_indices, item_indices):
        """
        Predict scores for given user-item pairs.
        :param user_indices: Tensor of user indices.
        :param item_indices: Tensor of item indices.
        :return: Predicted scores for the user-item pairs.
        """
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        return (user_emb * item_emb).sum(dim=1)


def calculate_bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """Bayesian Personalized Ranking (BPR) loss"""
    pos_scores = (user_emb * pos_item_emb).sum(dim=1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=1)
    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

    return loss


import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, coo_matrix
import warnings


class ItemKNN:
    def __init__(self, num_users, num_items, k=50, shrink=100, bm25_k1=1.2, bm25_b=0.75):
        """
        Weighted ItemKNN with BM25 feature weighting for sparse matrices.

        Args:
            num_users: Number of users
            num_items: Number of items
            k: Number of nearest neighbors
            shrink: Shrinkage parameter for similarity computation
            bm25_k1: BM25 k1 parameter (controls term frequency saturation)
            bm25_b: BM25 b parameter (controls length normalization)
        """
        self.num_users = num_users
        self.num_items = num_items
        self.k = k
        self.shrink = shrink
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.bm25_matrix = None
        self.similarity_matrix = None
        self.user_item_matrix = None

    def TF_IDF(self, rating_matrix):
        """
        Items are assumed to be on rows
        :param dataMatrix:
        :return:
        """

        # TFIDF each row of a sparse amtrix
        rating_matrix = sp.coo_matrix(rating_matrix)
        N = float(rating_matrix.shape[0])

        # calculate IDF
        idf = np.log(N / (1 + np.bincount(rating_matrix.col)))

        # apply TF-IDF adjustment
        rating_matrix.data = np.sqrt(rating_matrix.data) * idf[rating_matrix.col]

        return rating_matrix.tocsr().T

    def okapi_BM25(self, rating_matrix, K1=1.2, B=0.75):
        assert B > 0 and B < 1, "okapi_BM_25: B must be in (0,1)"
        assert K1 > 0, "okapi_BM_25: K1 must be > 0"

        # Weighs each row of a sparse matrix by OkapiBM25 weighting
        # calculate idf per term (user)

        rating_matrix = sp.coo_matrix(rating_matrix)

        N = float(rating_matrix.shape[0])
        idf = np.log(N / (1 + np.bincount(rating_matrix.col)))

        # calculate length_norm per document
        row_sums = np.ravel(rating_matrix.sum(axis=1))

        average_length = row_sums.mean()
        length_norm = (1.0 - B) + B * row_sums / average_length

        # weight matrix rows by bm25
        rating_matrix.data = rating_matrix.data * (K1 + 1.0) / (
                    K1 * length_norm[rating_matrix.row] + rating_matrix.data) * idf[rating_matrix.col]

        return rating_matrix.tocsr().T

    def cosine_sim_users(self, user_item_matrix):
        """
        Compute weighted cosine similarity between USERS (not items).
        """
        # No transpose needed - we want user-user similarities
        # user_item_matrix is already (users × items)

        # Compute norms for each user
        norms = np.sqrt(np.array(user_item_matrix.power(2).sum(axis=1)).flatten())
        norms[norms == 0] = 1.0

        # Normalize the matrix (each user vector)
        user_normalized = user_item_matrix.multiply(1 / norms[:, np.newaxis])

        # Compute similarity matrix (user × user)
        similarity = user_normalized @ user_normalized.T

        # Apply shrinkage based on common items (not users)
        if self.shrink > 0:
            # Count common items between users
            common_items = (user_item_matrix > 0).astype(float) @ (user_item_matrix > 0).T

            # Apply shrinkage formula
            common_items_coo = common_items.tocoo()
            shrink_data = common_items_coo.data / (common_items_coo.data + self.shrink)
            shrink_factor = csr_matrix(
                (shrink_data, (common_items_coo.row, common_items_coo.col)),
                shape=common_items.shape
            )
            similarity = similarity.multiply(shrink_factor)

        # Set diagonal to 0 (user shouldn't be similar to themselves)
        similarity.setdiag(0)

        return similarity.tocsr()

    def _recommend_items_user_similarities(self, user_ids, n_items):
        """
        Get top-n item recommendations using USER-based collaborative filtering.
        """
        recommendations = []

        for user_id in user_ids:
            # Get target user's profile
            target_user = self.tf_idf_matrix.getrow(user_id)
            user_item_indices = target_user.nonzero()[1]

            # Get similarities to all other users
            user_similarities = self.user_similarity_matrix.getrow(user_id)

            # Method 1: Use all similar users (if not pruned)
            if user_similarities.nnz > 0:
                # Compute item scores by aggregating similar users' preferences
                # user_similarities: (1, num_users)
                # self.tf_idf_matrix: (num_users, num_items)
                scores = (user_similarities @ self.tf_idf_matrix).toarray().flatten()
            else:
                # Fallback if no similar users found
                scores = np.zeros(self.num_items)

            # Remove items user already interacted with
            scores[user_item_indices] = -np.inf

            # Get top-n items
            if np.any(scores > -np.inf):
                top_items = np.argpartition(scores, -n_items)[-n_items:]
                top_items = top_items[np.argsort(scores[top_items])[::-1]]

                # Create recommendation list
                user_recs = [(item_id, scores[item_id]) for item_id in top_items
                             if scores[item_id] > -np.inf]
            else:
                user_recs = []

            recommendations.append(user_recs)

        return recommendations

    def cosine_sim(self, user_item_matrix):
        """
        Compute weighted cosine similarity between items using sparse operations.
        """
        item_user_matrix = user_item_matrix.T.tocsr()

        norms = np.sqrt(np.array(item_user_matrix.power(2).sum(axis=1)).flatten())

        norms[norms == 0] = 1.0

        # Normalize the matrix
        item_user_normalized = item_user_matrix.multiply(1 / norms[:, np.newaxis])

        # Compute similarity matrix (item x item)
        similarity = item_user_normalized @ item_user_normalized.T

        if self.shrink > 0:
            common_users = (item_user_matrix > 0).astype(float) @ (item_user_matrix > 0).T

            common_users_coo = common_users.tocoo()
            shrink_data = common_users_coo.data / (common_users_coo.data + self.shrink)
            shrink_factor = csr_matrix(
                (shrink_data, (common_users_coo.row, common_users_coo.col)),
                shape=common_users.shape
            )
            similarity = similarity.multiply(shrink_factor)

        # Set diagonal to 0 (item shouldn't be similar to itself)
        similarity.setdiag(0)

        return similarity.tocsr()

    def fit(self, interactions):
        """
        Fit the model on interaction data.

        Args:
            interactions: torch tensor of shape (nr_interactions, 3) with
                         columns [user_id, item_id, rating_weight]
        """
        # Convert torch tensor to numpy
        if isinstance(interactions, torch.Tensor):
            interactions_np = interactions.cpu().numpy()
        else:
            interactions_np = interactions

        # Create sparse user-item matrix
        self.user_item_matrix = csr_matrix(
            (interactions_np[:, 2].astype(float), (interactions_np[:, 0].astype(int), interactions_np[:, 1].astype(int))),
            shape=(self.num_users, self.num_items)
        )

        # Apply BM25 weighting
        # self.bm25_matrix = self.okapi_BM25(self.user_item_matrix.T)
        self.tf_idf_matrix = self.TF_IDF(self.user_item_matrix.T)

        # Compute weighted cosine similarity
        self.similarity_matrix = self.cosine_sim(self.tf_idf_matrix)

        # For each item, keep only top-k similar items
        self._prune_similarity_matrix()

    def _prune_similarity_matrix(self):
        """
        Keep only top-k similar items for each item to save memory.
        """
        rows = []
        cols = []
        data = []

        for item_idx in range(self.num_items):
            # Get similarities for this item
            item_similarities = self.similarity_matrix.getrow(item_idx).toarray().flatten()

            # Get top-k indices (excluding the item itself)
            top_k_indices = np.argpartition(item_similarities, -self.k)[-self.k:]
            top_k_indices = top_k_indices[item_similarities[top_k_indices] > 0]

            # Sort by similarity
            top_k_indices = top_k_indices[np.argsort(item_similarities[top_k_indices])[::-1]]

            # Add to sparse matrix data
            for neighbor_idx in top_k_indices:
                rows.append(item_idx)
                cols.append(neighbor_idx)
                data.append(item_similarities[neighbor_idx])

        # Create pruned similarity matrix
        self.similarity_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(self.num_items, self.num_items)
        )

    def predict(self, user_ids, item_ids=None, n_items=10):
        """
        Predict ratings or get top-n recommendations.

        Args:
            user_ids: User IDs to predict for (can be single ID or array)
            item_ids: Specific items to predict ratings for (optional)
            n_items: Number of items to recommend if item_ids is None

        Returns:
            If item_ids provided: predicted ratings
            Otherwise: top-n item recommendations with scores
        """
        if np.isscalar(user_ids):
            user_ids = [user_ids]

        if item_ids is not None:
            # Predict specific ratings
            return self._predict_ratings(user_ids, item_ids)
        else:
            # Get top-n recommendations
            return self._recommend_items_similarities(user_ids, n_items)

    def _predict_ratings(self, user_ids, item_ids):
        """
        Predict ratings for specific user-item pairs.
        """
        predictions = []

        for user_id in user_ids:
            # Get user's item ratings
            user_items = self.user_item_matrix.getrow(user_id)

            if isinstance(item_ids, (list, np.ndarray)):
                user_predictions = []
                for item_id in item_ids:
                    # Get similarities between target item and user's items
                    item_similarities = self.similarity_matrix.getrow(item_id)

                    # Compute weighted average
                    numerator = user_items.multiply(item_similarities).sum()
                    denominator = np.abs(item_similarities.toarray()).sum()

                    if denominator > 0:
                        prediction = numerator / denominator
                    else:
                        prediction = 0.0

                    user_predictions.append(prediction)
                predictions.append(user_predictions)
            else:
                # Single item prediction
                item_similarities = self.similarity_matrix.getrow(item_ids)
                numerator = user_items.multiply(item_similarities).sum()
                denominator = np.abs(item_similarities.toarray()).sum()

                if denominator > 0:
                    prediction = numerator / denominator
                else:
                    prediction = 0.0
                predictions.append(prediction)

        return np.array(predictions)

    def _recommend_items_similarities(self, user_ids, n_items):
        """
        Get top-n item recommendations for users using item-based collaborative filtering.
        """
        recommendations = []

        for user_id in user_ids:
            # Get user's BM25-weighted ratings
            user_items = self.tf_idf_matrix.getrow(user_id)
            user_item_indices = user_items.nonzero()[1]

            # Compute scores using item-item similarities
            # user_items: (1, num_items), similarity_matrix: (num_items, num_items)
            scores = (user_items @ self.similarity_matrix).toarray().flatten()

            # Remove items user already interacted with
            scores[user_item_indices] = -np.inf

            # Get top-n items - this part works correctly
            top_items = np.argpartition(scores, -n_items)[-n_items:]
            top_items = top_items[np.argsort(scores[top_items])[::-1]]

            # Create recommendation list
            user_recs = [(item_id, scores[item_id]) for item_id in top_items
                         if scores[item_id] > -np.inf]  # Changed condition
            recommendations.append(user_recs)

        return recommendations  # Remove np.array() wrapper


class MultiVAE(nn.Module):
    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


def multivae_loss(recon_batch, rating_weights, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(torch.nn.functional.log_softmax(recon_batch, 1) * rating_weights, dim=1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD


def get_model(dataset, config):
    """Initialize model based on type"""
    if config.model_name == 'LightGCN':
        return LightGCN(
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            embedding_dim=config.embedding_dim,
            num_layers=config.n_layers,
        ).to(config.device)
    elif config.model_name == 'ItemKNN':
        return ItemKNN(
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            k=config.item_knn_topk,
            shrink=config.shrink,
        )  # no .to(config.device) for ItemKNN as it uses numpy/scipy
    elif config.model_name == 'MultiVAE':
        return MultiVAE(
            p_dims=[config.latent_dimension, config.hidden_dimension, dataset.num_items],
            dropout=config.dropout_prob,
        ).to(config.device)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


