# Load libraries ---------------------------------------------

import pandas as pd

# ------------------------------------------------------------


class Recommender(object):
    """
    Base recommender class.
    """

    def __init__(self):
        """
        Initialize base recommender params and variables.
        """
        pass

    def fit(self, interactions_df, users_df, items_df):
        """
        Training of the recommender.

        :param pd.DataFrame interactions_df: DataFrame with recorded interactions between users and items
            defined by user_id, item_id and features of the interaction.
        :param pd.DataFrame users_df: DataFrame with users and their features defined by user_id
            and the user feature columns.
        :param pd.DataFrame items_df: DataFrame with items and their features defined by item_id
            and the item feature columns.
        """
        pass

    def recommend(self, users_df, items_df, n_recommendations=1):
        """
        Serving of recommendations. Scores items in items_df for each user in users_df and returns
        top n_recommendations for each user.

        :param pd.DataFrame users_df: DataFrame with users and their features for which recommendations
            should be generated.
        :param pd.DataFrame items_df: DataFrame with items and their features which should be scored.
        :param int n_recommendations: Number of recommendations to be returned for each user.
        :return: DataFrame with user_id, item_id and score as columns returning n_recommendations top recommendations
            for each user.
        :rtype: pd.DataFrame
        """

        del self, items_df

        recommendations = pd.DataFrame(columns=['user_id', 'item_id', 'score'])

        for ix, user in users_df.iterrows():
            user_recommendations = pd.DataFrame({'user_id': user['user_id'],
                                                 'item_id': [-1] * n_recommendations,
                                                 'score': [3.0] * n_recommendations})

            recommendations = pd.concat([recommendations, user_recommendations])

        return recommendations
