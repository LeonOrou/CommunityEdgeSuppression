import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from collections import defaultdict


def sparse_matrix_to_torch_sparse_tensor(X, device=None):
    """
    Convert a scipy sparse matrix to a torch sparse tensor, optionally moving it to the specified device.
    """
    coo = X.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
    values = torch.from_numpy(coo.data)
    shape = torch.Size(coo.shape)
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    
    if device is not None:
        sparse_tensor = sparse_tensor.to(device)
    
    return sparse_tensor


def build_adj_matrix_from_interactions(n_users, n_items, interactions, device=None):
    """
    Build adjacency matrix from interactions in (user, item, rating) format and convert to PyTorch sparse tensor.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        interactions: List of (user_id, item_id, rating) tuples
        device: PyTorch device to move the tensor to
        
    Returns:
        Normalized adjacency matrix as a PyTorch sparse tensor
    """
    # Extract user and item indices
    users = [int(x[0]) for x in interactions]
    items = [int(x[1]) for x in interactions]
    if len(interactions[0]) > 2:
        ratings = [float(x[2]) for x in interactions]
    else:
        ratings = [1.0] * len(interactions)
    
    # Remap item IDs to be after user IDs
    items_mapped = [i + n_users for i in items]
    
    # Total number of nodes
    n_nodes = n_users + n_items
    
    # Create adjacency matrix using scipy (just for construction)
    adj_mat = sp.coo_matrix(
        (ratings, (users, items_mapped)), 
        shape=(n_nodes, n_nodes)
    )
    
    # Make it symmetric
    adj_mat_sym = adj_mat + adj_mat.T
    
    # Normalize
    rowsum = np.array(adj_mat_sym.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    norm_adj = d_mat_inv_sqrt.dot(adj_mat_sym).dot(d_mat_inv_sqrt)
    
    # Convert to PyTorch sparse tensor
    return sparse_matrix_to_torch_sparse_tensor(norm_adj, device)


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, adj_matrix=None, interactions=None, embedding_size=256, n_layers=5, train_batch_size=256, eval_batch_size=256, device=None):
        super(LightGCN, self).__init__()
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        # Move embeddings to device
        self.user_embedding = self.user_embedding.to(self.device)
        self.item_embedding = self.item_embedding.to(self.device)
        
        # Handle adjacency matrix input
        if adj_matrix is not None:
            if isinstance(adj_matrix, sp.spmatrix):
                self.norm_adj_matrix = sparse_matrix_to_torch_sparse_tensor(adj_matrix, self.device)
            else:
                # Assume it's already a torch sparse tensor
                self.norm_adj_matrix = adj_matrix.to(self.device) if self.device else adj_matrix
        elif interactions is not None:
            self.norm_adj_matrix = build_adj_matrix_from_interactions(n_users, n_items, interactions, self.device)
        else:
            raise ValueError("Either adj_matrix or interactions must be provided")
        
    def _convert_sp_mat_to_tensor(self, X):
        """
        Convert sparse matrix to sparse tensor and move to device
        """
        return sparse_matrix_to_torch_sparse_tensor(X, self.device)
    
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
        # Ensure user_ids is on the correct device
        if isinstance(user_ids, torch.Tensor):
            user_ids = user_ids.to(self.device)
        else:
            user_ids = torch.tensor(user_ids, device=self.device)
            
        user_embeddings, item_embeddings = self.forward()
        user_emb = user_embeddings[user_ids]
        ratings = torch.matmul(user_emb, item_embeddings.T)
        return ratings


class ItemKNN:
    def __init__(self, n_users, n_items, k=250, shrink=10, device=None):
        # Hyperparameters
        self.k = k
        self.shrink = shrink
        
        # Model parameters
        self.n_users = n_users
        self.n_items = n_items
        self.similarity_matrix = None
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, train_matrix=None, interactions=None):
        """
        Calculate item similarity matrix using train_matrix or interactions
        """
        # Handle either matrix or interactions input
        if train_matrix is None and interactions is not None:
            # Convert interactions to user-item matrix
            if sp.issparse(train_matrix):
                train_matrix = train_matrix.tocsr()
            else:
                # Create from interactions
                users = [int(x[0]) for x in interactions]
                items = [int(x[1]) for x in interactions]
                ratings = [float(x[2]) if len(x) > 2 else 1.0 for x in interactions]
                train_matrix = sp.coo_matrix((ratings, (users, items)), shape=(self.n_users, self.n_items)).tocsr()
        elif train_matrix is None:
            raise ValueError("Either train_matrix or interactions must be provided")
            
        if sp.issparse(train_matrix):
            # Convert to torch tensors for GPU calculation
            coo = train_matrix.tocoo()
            values = torch.FloatTensor(coo.data)
            indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
            train_tensor = torch.sparse.FloatTensor(indices, values, 
                                                   torch.Size([self.n_users, self.n_items])).to(self.device)
            
            # For item-item matrix calculations, we'll materialize the dense tensor
            # Note: For large datasets, you may need to process in batches
            train_dense = train_tensor.to_dense()
            
            # Calculate item-item co-occurrence with PyTorch
            item_co_occurrence = torch.mm(train_dense.t(), train_dense)
            
            # Calculate item popularity (diagonal of co-occurrence matrix)
            item_popularity = torch.diag(item_co_occurrence)
            
            # Calculate similarity with shrinkage term
            similarity = item_co_occurrence.clone()
            
            # Apply shrinkage term to handle popularity bias
            for i in range(self.n_items):
                for j in range(self.n_items):
                    if i != j:
                        similarity[i, j] = similarity[i, j] / (similarity[i, j] + self.shrink)
            
            # Set diagonal to zero to avoid self-similarity
            similarity.fill_diagonal_(0)
            
            # Keep only top-k neighbors
            for i in range(self.n_items):
                # Get topk indices
                _, topk_indices = torch.topk(similarity[i], min(self.k, self.n_items))
                # Create mask of zeros
                mask = torch.zeros(self.n_items, device=self.device, dtype=torch.bool)
                # Set top-k indices to True
                mask[topk_indices] = True
                # Zero out non-top-k values
                similarity[i] = similarity[i] * mask.float()
            
            self.similarity_matrix = similarity
            
        else:
            # Handle case where train_matrix is already a dense matrix
            train_tensor = torch.tensor(train_matrix, dtype=torch.float, device=self.device)
            
            # Similar calculations as above
            item_co_occurrence = torch.mm(train_tensor.t(), train_tensor)
            # ... rest of the code similar to above
        
        return self
    
    def predict(self, user_ids, train_matrix=None, interactions=None):
        """
        Predict ratings for all items for each user in user_ids
        """
        # Handle either matrix or interactions input
        if train_matrix is None and interactions is not None:
            # Convert interactions to user-item matrix
            if sp.issparse(train_matrix):
                train_matrix = train_matrix.tocsr()
            else:
                # Create from interactions
                users = [int(x[0]) for x in interactions]
                items = [int(x[1]) for x in interactions]
                ratings = [float(x[2]) if len(x) > 2 else 1.0 for x in interactions]
                train_matrix = sp.coo_matrix((ratings, (users, items)), shape=(self.n_users, self.n_items)).tocsr()
        elif train_matrix is None:
            raise ValueError("Either train_matrix or interactions must be provided")
        
        # Ensure user_ids is a tensor on the correct device
        if isinstance(user_ids, torch.Tensor):
            user_ids = user_ids.to(self.device)
        else:
            user_ids = torch.tensor(user_ids, device=self.device)
        
        # Extract user interaction data
        if sp.issparse(train_matrix):
            # Convert sparse matrix rows to tensor
            user_data_list = []
            for user_id in user_ids.cpu().numpy():
                row = train_matrix[user_id].toarray().flatten()
                user_data_list.append(row)
            user_data = torch.tensor(np.vstack(user_data_list), dtype=torch.float, device=self.device)
        else:
            # Already a tensor or numpy array
            user_data = torch.tensor(train_matrix[user_ids.cpu().numpy()], dtype=torch.float, device=self.device)
        
        # Predict ratings using similarity matrix
        predictions = torch.mm(user_data, self.similarity_matrix)
        
        # Set already seen items to -inf (to avoid recommending them)
        for i, user_id in enumerate(user_ids.cpu().numpy()):
            if sp.issparse(train_matrix):
                user_items = train_matrix[user_id].indices
            else:
                user_items = np.where(train_matrix[user_id] > 0)[0]
            
            # Create mask for seen items
            mask = torch.zeros(self.n_items, device=self.device, dtype=torch.bool)
            mask[torch.tensor(user_items, device=self.device)] = True
            
            # Set seen items to -inf
            predictions[i][mask] = float('-inf')
            
        return predictions


