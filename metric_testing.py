import numpy as np


# prediction distributions
preds_uniform = [0.2, 0.2, 0.2, 0.2, 0.2]  # bias should be 0, min
preds_1 = [0.3, 0.25, 0.2, 0.15, 0.1]
preds_2 = [0.3, 0.3, 0.25, 0.1, 0.05]
preds_3 = [0.4, 0.4, 0.1, 0.05, 0.05]
preds_4 = [0.4, 0.3, 0.2, 0.1, 0.1]
preds_5 = [0.5, 0.3, 0.1, 0.1, 0.0]
preds_6 = [0.6, 0.2, 0.1, 0.1, 0.0]
preds_7 = [0.5, 0.5, 0.0, 0.0, 0.0]  # only two classes but fully, argubly more biased than preds_6, also metric says it
preds_8 = [0.7, 0.2, 0.05, 0.025, 0.025]
preds_9 = [0.8, 0.1, 0.05, 0.025, 0.025]
preds_10 = [0.9, 0.05, 0.025, 0.025, 0.0]
preds_11 = [1.0, 0.0, 0.0, 0.0, 0.0]  # bias should be 1, max


def squared_sum(preds_uniform, preds_biased):
    preds_uniform = np.array(preds_uniform)
    preds_biased = np.array(preds_biased)
    ss_opt = np.sum(preds_uniform ** 2)
    ss = np.sum(preds_biased ** 2)
    ss_normalized = (np.abs(ss - ss_opt))/(1 - ss_opt)  # normalize to 1
    return ss_normalized


def lp_distance(preds_uniform, preds_worst, preds_biased, p=2):
    preds_uniform = np.array(preds_uniform)
    preds_worst = np.array(preds_worst)
    preds_biased = np.sort(np.array(preds_biased))
    lp = np.sum(np.abs(preds_uniform - preds_biased) ** p) ** (1 / p)
    lp_worst = np.sum(np.abs(preds_uniform - preds_worst) ** p) ** (1 / p)
    lp_normalized = lp / lp_worst  # normalize to 1
    return lp_normalized


# comparing all distributions with all measures
preds = [preds_uniform, preds_1, preds_2, preds_3, preds_4, preds_5, preds_6, preds_7, preds_8, preds_9, preds_10, preds_11]

mse_results = []
lp_distance_results = []

for i in range(len(preds)):
    mse_results.append(squared_sum(preds[0], preds[i]))
    lp_distance_results.append(lp_distance(preds_uniform=preds[0], preds_worst=preds[-1], preds_biased=preds[i]))
    print(f"Distribution {i}:")
    print(f"  MSE: {mse_results[i]}")
    print(f"  Lp Distance: {lp_distance_results[i]}")




