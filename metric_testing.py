import numpy as np


# prediction distributions
preds_uniform = [0.2, 0.2, 0.2, 0.2, 0.2]  # bias should be 0
preds_1 = [0.3, 0.25, 0.2, 0.15, 0.1]  #
preds_2 = [0.4, 0.3, 0.2, 0.1, 0.1]  #
preds_3 = [0.5, 0.3, 0.1, 0.1, 0.0]  #
preds_4 = [0.6, 0.2, 0.1, 0.1, 0.0]  #
preds_5 = [0.3, 0.3, 0.25, 0.1, 0.05]
preds_6 = [0.4, 0.4, 0.1, 0.05, 0.05]  #
preds_7 = [0.7, 0.2, 0.05, 0.025, 0.025]  #
preds_8 = [0.8, 0.1, 0.05, 0.025, 0.025]  #
preds_8 = [0.9, 0.05, 0.025, 0.025, 0.025]  # 
preds_9 = [1.0, 0.0, 0.0, 0.0, 0.0]  # bias should be 1

def mean_squared(preds_uniform, preds_b, one_minus_x=False, subtract_optimum=True):
    """
    Calculate the mean squared error between two probability distributions.
    :param subtract_optimum: if true, subtract uniform optimum and normalize to 1
    """
    preds_uniform = np.array(preds_uniform)
    preds_b = np.array(preds_b)
    max_value = 1
    if one_minus_x:
        preds_uniform = 1 - preds_uniform
        preds_b = 1 - preds_b
    ms_opt = np.sum(preds_uniform ** 2)
    if subtract_optimum:
        max_value = ms_opt * len(preds_uniform)
    ms = np.sum(preds_b ** 2)
    if subtract_optimum:
        ms = (np.abs(ms - ms_opt))/(1 - ms_opt)
    return ms

def kl_divergence(preds_uniform, preds_b):
    """
    Calculate the Kullback-Leibler divergence between two probability distributions.
    """
    preds_uniform = np.array(preds_uniform)
    preds_b = np.array(preds_b)
    kl_div = np.sum(preds_uniform * np.log(preds_uniform / preds_b))
    return kl_div

# comparing all distributions with all measures
preds = [preds_uniform, preds_1, preds_2, preds_3, preds_4, preds_5, preds_6, preds_7, preds_8, preds_9]
mse_results = []
mase_one_minus_results = []
kl_results = []

for i in range(len(preds)):
    mse_results.append(mean_squared(preds_uniform, preds[i]))
    mase_one_minus_results.append(mean_squared(preds_uniform, preds[i], one_minus_x=True))
    kl_results.append(kl_divergence(preds_uniform, preds[i]))
    print(f"Distribution {i}:")
    print(f"  MSE: {mse_results[i]}")
    print(f"  MSE (1-x): {mase_one_minus_results[i]}")
    print(f"  KL Divergence: {kl_results[i]}")




