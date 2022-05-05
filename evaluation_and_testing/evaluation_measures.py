# Load libraries ---------------------------------------------

import numpy as np
import pandas as pd
from collections import defaultdict

# ------------------------------------------------------------


def rmse(r_pred, r_real):
    return np.sqrt(np.sum(np.power(r_pred - r_real, 2)) / len(r_pred))


def mre(r_pred, r_real):
    return 1 / len(r_pred) * np.sum(np.abs(r_pred - r_real) / np.abs(r_real))


def mape(r_pred, r_real):
    return mre(r_pred, r_real)


def tre(r_pred, r_real):
    return np.sum(np.abs(r_pred - r_real)) / np.sum(np.abs(r_real))


def hr(recommendations, real_interactions, n=1):
    """
    Assumes recommendations are ordered by user_id and then by score.

    :param pd.DataFrame recommendations:
    :param pd.DataFrame real_interactions:
    :param int n:
    """
    # Transform real_interactions to a dict for a large speed-up
    rui = defaultdict(lambda: 0)

    for idx, row in real_interactions.iterrows():
        rui[(row['user_id'], row['item_id'])] = 1

    result = 0.0

    previous_user_id = -1
    rank = 0
    for idx, row in recommendations.iterrows():
        if previous_user_id == row['user_id']:
            rank += 1
        else:
            rank = 1

        if rank <= n:
            result += rui[(row['user_id'], row['item_id'])]

        previous_user_id = row['user_id']

    if len(recommendations['user_id'].unique()) > 0:
        result /= len(recommendations['user_id'].unique())

    return result


def ndcg(recommendations, real_interactions, n=1):
    """
    Assumes recommendations are ordered by user_id and then by score.

    :param pd.DataFrame recommendations:
    :param pd.DataFrame real_interactions:
    :param int n:
    """
    # Transform real_interactions to a dict for a large speed-up
    rui = defaultdict(lambda: 0)

    for idx, row in real_interactions.iterrows():
        rui[(row['user_id'], row['item_id'])] = 1

    result = 0.0

    previous_user_id = -1
    rank = 0
    for idx, row in recommendations.iterrows():
        if previous_user_id == row['user_id']:
            rank += 1
        else:
            rank = 1

        if rank <= n:
            result += rui[(row['user_id'], row['item_id'])] / np.log2(1 + rank)

        previous_user_id = row['user_id']

    if len(recommendations['user_id'].unique()) > 0:
        result /= len(recommendations['user_id'].unique())

    return result
