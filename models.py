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
        self.similarity_matrix = None
        self.user_item_matrix = None

    def _compute_bm25_weights(self, user_item_matrix):
        """
        Compute BM25 weights for the user-item matrix.
        """
        # Convert to CSR for efficient operations
        if not isinstance(user_item_matrix, csr_matrix):
            user_item_matrix = user_item_matrix.tocsr()

        # Number of items each user has interacted with (document lengths)
        item_lengths = np.array(user_item_matrix.sum(axis=0)).flatten()
        avg_length = item_lengths.mean()

        # Number of users who interacted with each item (document frequencies)
        df = np.array((user_item_matrix > 0).sum(axis=0)).flatten()

        # Compute IDF
        idf = np.log((self.num_users - df + 0.5) / (df + 0.5) + 1.0)

        # Create BM25 weighted matrix
        rows, cols = user_item_matrix.nonzero()
        data = user_item_matrix.data

        # Compute BM25 scores
        bm25_scores = []
        for idx, (user, item, weight) in enumerate(zip(rows, cols, data)):
            tf = weight
            doc_len = item_lengths[item]

            # BM25 formula
            numerator = idf[item] * tf * (self.bm25_k1 + 1)
            denominator = tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_len / avg_length)
            bm25_score = numerator / denominator
            bm25_scores.append(bm25_score)

        bm25_matrix = csr_matrix((bm25_scores, (rows, cols)), shape=user_item_matrix.shape)
        return bm25_matrix

    def _weighted_cosine_similarity_sparse(self, bm25_matrix):
        """
        Compute weighted cosine similarity between items using sparse operations.
        """
        # Transpose to get item-user matrix
        item_user_matrix = bm25_matrix.T.tocsr()

        # Compute norms for each item
        norms = np.sqrt(np.array(item_user_matrix.power(2).sum(axis=1)).flatten())

        # Avoid division by zero
        norms[norms == 0] = 1.0

        # Normalize the matrix
        item_user_normalized = item_user_matrix.multiply(1 / norms[:, np.newaxis])

        # Compute similarity matrix (item x item)
        similarity = item_user_normalized @ item_user_normalized.T

        # Apply shrinkage
        if self.shrink > 0:
            # Get the number of common users between items
            common_users = (item_user_matrix > 0).astype(float) @ (item_user_matrix > 0).T

            # Apply shrinkage formula
            # Convert to COO format for efficient element-wise operations
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

        users = interactions_np[:, 0].astype(int)
        items = interactions_np[:, 1].astype(int)
        weights = interactions_np[:, 2].astype(float)

        # Create sparse user-item matrix
        self.user_item_matrix = csr_matrix(
            (weights, (users, items)),
            shape=(self.num_users, self.num_items)
        )

        # Apply BM25 weighting
        bm25_matrix = self._compute_bm25_weights(self.user_item_matrix)

        # Compute weighted cosine similarity
        self.similarity_matrix = self._weighted_cosine_similarity_sparse(bm25_matrix)

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
            return self._recommend_items(user_ids, n_items)

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

    def _recommend_items(self, user_ids, n_items):
        """
        Get top-n item recommendations for users.
        """
        recommendations = []

        for user_id in user_ids:
            # Get user's item ratings
            user_items = self.user_item_matrix.getrow(user_id)
            user_item_indices = user_items.nonzero()[1]

            # Compute scores for all items
            # scores = (user_items @ self.similarity_matrix).toarray().flatten()
            # scores = scores.toarray().flatten()
            #
            # # Normalize by sum of similarities
            # sim_sums = np.array(self.similarity_matrix.sum(axis=0)).flatten()
            # sim_sums[sim_sums == 0] = 1.0
            # scores = scores / sim_sums
            item_similarities = self.similarity_matrix[:, user_item_indices]
            scores = np.array(item_similarities.sum(axis=1)).flatten()

            # Remove items user already interacted with
            scores[user_item_indices] = -np.inf

            # Get top-n items
            top_items = np.argpartition(scores, -n_items)[-n_items:]
            top_items = top_items[np.argsort(scores[top_items])[::-1]]

            # Create recommendation list
            user_recs = [(item_id, scores[item_id]) for item_id in top_items if scores[item_id] > 0]
            recommendations.append(user_recs)

        return np.array(recommendations)

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


