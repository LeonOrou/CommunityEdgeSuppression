import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from collections import defaultdict


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, adj_matrix, embedding_size=256, n_layers=5, train_batch_size=256, eval_batch_size=256):
        super(LightGCN, self).__init__()
        
        # Hyperparameters
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        
        # Model parameters
        self.n_users = n_users
        self.n_items = n_items
        
        # Embedding layers for users and items
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.item_embedding = nn.Embedding(n_items, embedding_size)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # Normalize adjacency matrix for graph convolution
        self.norm_adj_matrix = self._convert_sp_mat_to_tensor(adj_matrix)
        
    def _convert_sp_mat_to_tensor(self, X):
        """
        Convert sparse matrix to sparse tensor
        """
        coo = X.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        values = torch.from_numpy(coo.data)
        shape = torch.Size(coo.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def forward(self, users=None, items=None):
        # Get initial user and item embeddings
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        
        # Storage for embeddings from each layer
        embeddings_list = [all_embeddings]
        
        # Graph convolution layers
        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        # Sum of embeddings from all layers (with equal weighting)
        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        # Split user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items])
        
        if users is not None and items is not None:
            # For batch prediction
            user_emb = user_all_embeddings[users]
            item_emb = item_all_embeddings[items]
            return user_emb, item_emb
        else:
            # Return all embeddings
            return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, users, pos_items, neg_items):
        user_emb, pos_item_emb = self.forward(users, pos_items)
        _, neg_item_emb = self.forward(users, neg_items)
        
        # BPR loss
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # L2 regularization
        reg_loss = 1/2 * (user_emb.norm(2).pow(2) + 
                          pos_item_emb.norm(2).pow(2) + 
                          neg_item_emb.norm(2).pow(2)) / len(users)
        
        return loss + 1e-4 * reg_loss
    
    def predict(self, user_ids):
        user_embeddings, item_embeddings = self.forward()
        user_emb = user_embeddings[user_ids]
        ratings = torch.matmul(user_emb, item_embeddings.T)
        return ratings


class ItemKNN:
    def __init__(self, n_users, n_items, k=250, shrink=10):
        # Hyperparameters
        self.k = k
        self.shrink = shrink
        
        # Model parameters
        self.n_users = n_users
        self.n_items = n_items
        self.similarity_matrix = None
        
    def fit(self, train_matrix):
        """
        Calculate item similarity matrix using train_matrix (user-item interaction matrix)
        """
        # Convert to scipy sparse matrix if it's not already
        if not sp.issparse(train_matrix):
            train_matrix = sp.csr_matrix(train_matrix)
        
        # Calculate item-item co-occurrence
        item_co_occurrence = train_matrix.T @ train_matrix
        
        # Calculate item popularity (diagonal of co-occurrence matrix)
        item_popularity = np.diag(item_co_occurrence.toarray())
        
        # Calculate similarity with shrinkage term
        similarity = item_co_occurrence.toarray().copy()
        
        # Apply shrinkage term to handle popularity bias
        for i in range(self.n_items):
            for j in range(self.n_items):
                if i != j:
                    similarity[i, j] = similarity[i, j] / (similarity[i, j] + self.shrink)
        
        # Set diagonal to zero to avoid self-similarity
        np.fill_diagonal(similarity, 0)
        
        # Keep only top-k neighbors
        for i in range(self.n_items):
            sorted_indices = np.argsort(similarity[i])[::-1]
            if len(sorted_indices) > self.k:
                # Set similarity to 0 for all except top-k
                mask = np.ones(self.n_items, dtype=bool)
                mask[sorted_indices[:self.k]] = False
                similarity[i, mask] = 0
        
        self.similarity_matrix = similarity
        return self
    
    def predict(self, user_ids, train_matrix):
        """
        Predict ratings for all items for each user in user_ids
        """
        if not sp.issparse(train_matrix):
            train_matrix = sp.csr_matrix(train_matrix)
        
        # Extract user interaction data
        user_data = train_matrix[user_ids]
        
        # Predict ratings using similarity matrix
        predictions = user_data @ self.similarity_matrix
        
        # Convert to dense array for convenient handling
        if sp.issparse(predictions):
            predictions = predictions.toarray()
        
        # Set already seen items to -inf (to avoid recommending them)
        for i, user_id in enumerate(user_ids):
            user_items = train_matrix[user_id].indices
            predictions[i, user_items] = -np.inf
            
        return torch.from_numpy(predictions)


class MultiVAE(nn.Module):
    def __init__(self, n_items, hidden_dimension=800,
                 latent_dimension=200,
                 dropout_prob=0.7,
                 train_batch_size=4096,
                 eval_batch_size=4096,
                 anneal_cap=0.3,
                 total_anneal_steps=200000):
        super(MultiVAE, self).__init__()
        
        # Hyperparameters
        self.hidden_dimension = hidden_dimension
        self.latent_dimension = latent_dimension
        self.dropout_prob = dropout_prob
        self.anneal_cap = anneal_cap
        self.total_anneal_steps = total_anneal_steps
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        
        # Track training updates for annealing
        self.update_count = 0
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(n_items, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.Tanh()
        )
        
        # Latent space projections
        self.mu = nn.Linear(hidden_dimension, latent_dimension)
        self.logvar = nn.Linear(hidden_dimension, latent_dimension)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dimension, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, n_items)
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
    def encode(self, x):
        h = self.dropout(self.encoder(x))
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def calculate_loss(self, x):
        # Forward pass
        x_pred, mu, logvar = self.forward(x)
        
        # Calculate reconstruction loss
        recon_loss = -torch.sum(F.log_softmax(x_pred, dim=1) * x, dim=1).mean()
        
        # Calculate KL divergence
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        # Apply KL annealing
        self.update_count += 1
        anneal = min(self.anneal_cap, self.update_count / self.total_anneal_steps)
        
        # Total loss
        loss = recon_loss + anneal * kl_div
        
        return loss
    
    def predict(self, user_ids, train_matrix):
        """
        Generate predictions for a batch of users
        """
        # Extract user interaction data
        user_data = train_matrix[user_ids]
        
        # Convert to tensor if needed
        if not isinstance(user_data, torch.Tensor):
            if sp.issparse(user_data):
                user_data = torch.FloatTensor(user_data.toarray())
            else:
                user_data = torch.FloatTensor(user_data)
        
        # Set model to eval mode
        self.eval()
        
        # Get predictions
        with torch.no_grad():
            predictions, _, _ = self.forward(user_data)
            
        # Ensure already seen items are not recommended
        predictions_np = predictions.cpu().numpy()
        for i, user_id in enumerate(user_ids):
            seen_items = train_matrix[user_id].indices if sp.issparse(train_matrix) else np.where(train_matrix[user_id] > 0)[0]
            predictions_np[i, seen_items] = -np.inf
            
        return torch.from_numpy(predictions_np)
