import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.sparse as sp
from models.ItemKNN import ItemKNN
from models.LightGCN import LightGCN
from models.MultVAE import MultVAE
from utils import power_node_edge_dropout, set_seed

# Fixed hyperparameters based on research
FIXED_K = 50
FIXED_SHRINK = 10.0
FIXED_NUM_LAYERS = 3
FIXED_LEARNING_RATE = 0.01
FIXED_EMBEDDING_DIM = 64

set_seed(42)

def objective(trial, train_path, test_path):
    X = np.load(train_path)
    y = np.load(test_path)
    X = X.tocsr()
    y = y.tocsr()

    X_dropped = power_node_edge_dropout(X, **trial.params)

    # depending on model, different input is needed
    if trial.suggest_categorical('model', ['MultVAE', 'ItemKNN', 'LightGCN']) == 'MultVAE':
        model = MultVAE()
    elif trial.suggest_categorical('model', ['MultVAE', 'ItemKNN', 'LightGCN']) == 'ItemKNN':
        model = ItemKNN(k=FIXED_K, shrink=FIXED_SHRINK)
    else:  # LightGCN
        model = LightGCN(num_layers=FIXED_NUM_LAYERS, learning_rate=FIXED_LEARNING_RATE, embedding_dim=FIXED_EMBEDDING_DIM

    model.fit(X_dropped)
    predictions = model.predict(y)
    accuracy = accuracy_score(y.nonzero()[1], predictions)
    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print("Best hyperparameters for power_node_edge_dropout:", study.best_params)