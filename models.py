import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing



# Custom LightGCN implementation with edge weights
class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__(aggr='add')
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, edge_index, edge_weight=None):
        # Get initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        # Combine user and item embeddings
        x = torch.cat([user_emb, item_emb], dim=0)

        # Store embeddings from each layer
        embeddings = [x]

        for _ in range(self.num_layers):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            embeddings.append(x)

        # Average embeddings across all layers
        final_embedding = torch.stack(embeddings, dim=0).mean(dim=0)

        # Split back into user and item embeddings
        user_final = final_embedding[:self.num_users]
        item_final = final_embedding[self.num_users:]

        return user_final, item_final

    def message(self, x_j, edge_weight):
        # Apply edge weights to messages
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out, x):
        return aggr_out

