import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from RecSys_PyTorch.models.BaseModel import BaseModel
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import degree


# Custom LightGCN implementation with edge weights
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
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


class ItemKNN(nn.Module):
    def __init__(self, dataset, hparams, device):
        super(ItemKNN, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.topk = hparams['topk']
        self.shrink = hparams['shrink']
        self.feature_weighting = hparams['feature_weighting']
        assert self.feature_weighting in ['tf-idf', 'bm25', 'none']

    def fit_knn(self, train_matrix, block_size=500):
        if self.feature_weighting == 'tf-idf':
            train_matrix = self.TF_IDF(train_matrix.T).T
        elif self.feature_weighting == 'bm25':
            train_matrix = self.okapi_BM25(train_matrix.T).T

        train_matrix = train_matrix.tocsc()
        num_items = train_matrix.shape[1]

        start_col_local = 0
        end_col_local = num_items

        start_col_block = start_col_local

        this_block_size = 0
        block_size = 500

        sumOfSquared = np.array(train_matrix.power(2).sum(axis=0)).ravel()
        sumOfSquared = np.sqrt(sumOfSquared)

        values = []
        rows = []
        cols = []
        while start_col_block < end_col_local:
            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block - start_col_block

            # All data points for a given item
            # item_data: user, item blocks
            item_data = train_matrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray().squeeze()

            # If only 1 feature avoid last dimension to disappear
            if item_data.ndim == 1:
                item_data = np.atleast_2d(item_data)

            this_block_weights = train_matrix.T.dot(item_data)

            for col_index_in_block in range(this_block_size):
                # this_block_size: (item,)
                # similarity between 'one block item' and whole items
                if this_block_size == 1:
                    this_column_weights = this_block_weights
                else:
                    this_column_weights = this_block_weights[:, col_index_in_block]

                # columnIndex = item index
                # zero out self similarity
                columnIndex = col_index_in_block + start_col_block
                this_column_weights[columnIndex] = 0.0

                # cosine similarity
                # denominator = sqrt(l2_norm(x)) * sqrt(l2_norm(y))+ shrinkage + eps
                denominator = sumOfSquared[columnIndex] * sumOfSquared + self.shrink + 1e-6
                this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                relevant_items_partition = (-this_column_weights).argpartition(self.topk - 1)[0:self.topk]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_column_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_column_weights[top_k_idx][notZerosMask])
                rows.extend(top_k_idx[notZerosMask])
                cols.extend(np.ones(numNotZeros) * columnIndex)

            start_col_block += block_size

        self.W_sparse = sp.csr_matrix((values, (rows, cols)),
                                      shape=(num_items, num_items),
                                      dtype=np.float32)

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        train_matrix = dataset.train_data
        self.fit_knn(train_matrix)

        output = train_matrix @ self.W_sparse

        loss = F.binary_cross_entropy(torch.tensor(train_matrix.toarray()), torch.tensor(output.toarray()))

        if evaluator is not None:
            scores = evaluator.evaluate(self)
        else:
            scores = None

        if loggers is not None:
            if evaluator is not None:
                for logger in loggers:
                    logger.log_metrics(scores, epoch=1)

        return {'scores': scores, 'loss': loss}

    def predict(self, eval_users, eval_pos, test_batch_size):
        input_matrix = eval_pos.toarray()
        preds = np.zeros_like(input_matrix)

        num_data = input_matrix.shape[0]
        num_batches = int(np.ceil(num_data / test_batch_size))
        perm = list(range(num_data))
        for b in range(num_batches):
            if (b + 1) * test_batch_size >= num_data:
                batch_idx = perm[b * test_batch_size:]
            else:
                batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
            test_batch_matrix = input_matrix[batch_idx]
            batch_pred_matrix = (test_batch_matrix @ self.W_sparse)
            preds[batch_idx] = batch_pred_matrix

        preds[eval_pos.nonzero()] = float('-inf')

        return preds

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

        return rating_matrix.tocsr()

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

        return rating_matrix.tocsr()


