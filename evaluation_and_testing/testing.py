# Load libraries ---------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from evaluation_and_testing.evaluation_measures import rmse
from evaluation_and_testing.evaluation_measures import mape
from evaluation_and_testing.evaluation_measures import tre
from evaluation_and_testing.evaluation_measures import hr
from evaluation_and_testing.evaluation_measures import ndcg

# ------------------------------------------------------------


def evaluate_train_test_split_explicit(recommender, interactions_df, items_df, seed=6789):
    rng = np.random.RandomState(seed=seed)

    if isinstance(interactions_df, dict):
        # If interactions_df is a dict with already split data, use the split
        interactions_df_train = interactions_df['train']
        interactions_df_test = interactions_df['test']
    else:
        # Otherwise split the dataset into train and test

        shuffle = np.arange(len(interactions_df))
        rng.shuffle(shuffle)
        shuffle = list(shuffle)

        train_test_split = 0.8
        split_index = int(len(interactions_df) * train_test_split)

        interactions_df_train = interactions_df.iloc[shuffle[:split_index]]
        interactions_df_test = interactions_df.iloc[shuffle[split_index:]]

    # Train the recommender

    recommender.fit(interactions_df_train, None, items_df)

    # Gather predictions

    r_pred = []

    for idx, row in interactions_df_test.iterrows():
        users_df = pd.DataFrame([row['user_id']], columns=['user_id'])
        eval_items_df = pd.DataFrame([row['item_id']], columns=['item_id'])
        eval_items_df = pd.merge(eval_items_df, items_df, on='item_id')
        recommendations = recommender.recommend(users_df, eval_items_df, n_recommendations=1)

        r_pred.append(recommendations.iloc[0]['score'])

    # Gather real ratings

    r_real = np.array(interactions_df_test['rating'].tolist())

    # Return evaluation metrics

    return rmse(r_pred, r_real), mape(r_pred, r_real), tre(r_pred, r_real)


def evaluate_train_test_split_implicit(recommender, interactions_df, items_df, seed=6789):
    # Write your code here
    rng = np.random.RandomState(seed=seed)

    if isinstance(interactions_df, dict):
        # If interactions_df is a dict with already split data, use the split
        interactions_df_train = interactions_df['train']
        interactions_df_test = interactions_df['test']
    else:
        # Otherwise split the dataset into train and test

        shuffle = np.arange(len(interactions_df))
        rng.shuffle(shuffle)
        shuffle = list(shuffle)

        train_test_split = 0.8
        split_index = int(len(interactions_df) * train_test_split)

        interactions_df_train = interactions_df.iloc[shuffle[:split_index]]
        interactions_df_test = interactions_df.iloc[shuffle[split_index:]]

    hr_1 = []
    hr_3 = []
    hr_5 = []
    hr_10 = []
    ndcg_1 = []
    ndcg_3 = []
    ndcg_5 = []
    ndcg_10 = []

    # Train the recommender

    recommender.fit(interactions_df_train, None, items_df)

    # Make recommendations for each user in the test set and calculate the metric
    # against all items of that user in the test set

    test_user_interactions = interactions_df_test.groupby(by='user_id')

    for user_id, user_interactions in test_user_interactions:

        recommendations = recommender.recommend(pd.DataFrame([user_id], columns=['user_id']),
                                                items_df, n_recommendations=10)

        hr_1.append(hr(recommendations, user_interactions, n=1))
        hr_3.append(hr(recommendations, user_interactions, n=3))
        hr_5.append(hr(recommendations, user_interactions, n=5))
        hr_10.append(hr(recommendations, user_interactions, n=10))
        ndcg_1.append(ndcg(recommendations, user_interactions, n=1))
        ndcg_3.append(ndcg(recommendations, user_interactions, n=3))
        ndcg_5.append(ndcg(recommendations, user_interactions, n=5))
        ndcg_10.append(ndcg(recommendations, user_interactions, n=10))

    hr_1 = np.mean(hr_1)
    hr_3 = np.mean(hr_3)
    hr_5 = np.mean(hr_5)
    hr_10 = np.mean(hr_10)
    ndcg_1 = np.mean(ndcg_1)
    ndcg_3 = np.mean(ndcg_3)
    ndcg_5 = np.mean(ndcg_5)
    ndcg_10 = np.mean(ndcg_10)

    return hr_1, hr_3, hr_5, hr_10, ndcg_1, ndcg_3, ndcg_5, ndcg_10


def evaluate_leave_one_out_explicit(recommender, interactions_df, items_df, max_evals=300, seed=6789):
    rng = np.random.RandomState(seed=seed)

    # Prepare splits of the datasets
    kf = KFold(n_splits=len(interactions_df), random_state=rng, shuffle=True)

    # For each split of the dataset train the recommender, generate recommendations and evaluate

    r_pred = []
    r_real = []
    n_eval = 1
    for train_index, test_index in kf.split(interactions_df.index):
        interactions_df_train = interactions_df.loc[interactions_df.index[train_index]]
        interactions_df_test = interactions_df.loc[interactions_df.index[test_index]]

        recommender.fit(interactions_df_train, None, items_df)
        recommendations = recommender.recommend(
            interactions_df_test.loc[:, ['user_id']],
            items_df.loc[items_df['item_id'] == interactions_df_test.iloc[0]['item_id']])

        r_pred.append(recommendations.iloc[0]['score'])
        r_real.append(interactions_df_test.iloc[0]['rating'])

        if n_eval == max_evals:
            break
        n_eval += 1

    r_pred = np.array(r_pred)
    r_real = np.array(r_real)

    # Return evaluation metrics

    return rmse(r_pred, r_real), mape(r_pred, r_real), tre(r_pred, r_real)


def evaluate_leave_one_out_implicit(recommender, interactions_df, items_df, max_evals=300, seed=6789):
    rng = np.random.RandomState(seed=seed)

    # Prepare splits of the datasets
    kf = KFold(n_splits=len(interactions_df), random_state=rng, shuffle=True)

    hr_1 = []
    hr_3 = []
    hr_5 = []
    hr_10 = []
    ndcg_1 = []
    ndcg_3 = []
    ndcg_5 = []
    ndcg_10 = []

    # For each split of the dataset train the recommender, generate recommendations and evaluate

    n_eval = 1
    for train_index, test_index in kf.split(interactions_df.index):
        interactions_df_train = interactions_df.loc[interactions_df.index[train_index]]
        interactions_df_test = interactions_df.loc[interactions_df.index[test_index]]

        recommender.fit(interactions_df_train, None, items_df)
        recommendations = recommender.recommend(
            interactions_df_test.loc[:, ['user_id']], items_df, n_recommendations=10)

        hr_1.append(hr(recommendations, interactions_df_test, n=1))
        hr_3.append(hr(recommendations, interactions_df_test, n=3))
        hr_5.append(hr(recommendations, interactions_df_test, n=5))
        hr_10.append(hr(recommendations, interactions_df_test, n=10))
        ndcg_1.append(ndcg(recommendations, interactions_df_test, n=1))
        ndcg_3.append(ndcg(recommendations, interactions_df_test, n=3))
        ndcg_5.append(ndcg(recommendations, interactions_df_test, n=5))
        ndcg_10.append(ndcg(recommendations, interactions_df_test, n=10))

        if n_eval == max_evals:
            break
        n_eval += 1

    hr_1 = np.mean(hr_1)
    hr_3 = np.mean(hr_3)
    hr_5 = np.mean(hr_5)
    hr_10 = np.mean(hr_10)
    ndcg_1 = np.mean(ndcg_1)
    ndcg_3 = np.mean(ndcg_3)
    ndcg_5 = np.mean(ndcg_5)
    ndcg_10 = np.mean(ndcg_10)

    return hr_1, hr_3, hr_5, hr_10, ndcg_1, ndcg_3, ndcg_5, ndcg_10
