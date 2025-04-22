import pandas as pd
from sklearn.model_selection import train_test_split
from utils_functions import set_seed

set_seed(42)


def threshold_data(data_pd, threshold=4, rating_col_name='rating'):
    # making data above threshold 1 and below 0
    data_pd[rating_col_name] = data_pd[rating_col_name].apply(lambda x: 1 if x >= threshold else 0)
    # dropping rows where 0 ratings
    data_pd = data_pd[data_pd[rating_col_name] == 1]
    return data_pd


def booleanify(data_pd, threshold=4, rating_col_name='rating'):
    # although RecBole does threshold the data, in case it doesn't cut the remaining edges we do it here too instead setting all rating to 1
    data_pd[rating_col_name] = data_pd[rating_col_name].apply(lambda x: 1 if x >= threshold else 0)
    # dropping rows where 0 ratings
    data_pd = data_pd[data_pd[rating_col_name] == 1]
    return data_pd


def threshold_degree(data_pd, threshold=5, user_col_name='userId'):
    node_degrees = data_pd.groupby(user_col_name).size()
    # keep only users with at least threshold degrees
    data_pd = data_pd[data_pd[user_col_name].apply(lambda x: node_degrees[x] >= threshold)]
    return data_pd



