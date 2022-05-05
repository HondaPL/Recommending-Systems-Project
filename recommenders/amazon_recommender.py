# Load libraries ---------------------------------------------

import pandas as pd
import numpy as np
import scipy.special as scisp

from recommenders.recommender import Recommender

# ------------------------------------------------------------


class AmazonRecommender(Recommender):
    """
    Basic item-to-item collaborative filtering algorithm used in Amazon.com as described in:
    - Linden G., Smith B., York Y., Amazon.com Recommendations. Item-to-Item Collaborative Filtering,
        IEEE Internet Computing, 2003,
    - Smith B., Linden G., Two Decades of Recommender Systems at Amazon.com, IEEE Internet Computing, 2017.
    """

    def __init__(self):
        super().__init__()
        self.recommender_df = pd.DataFrame(columns=['user_id', 'item_id', 'score'])
        self.interactions_df = None
        self.item_id_mapping = None
        self.user_id_mapping = None
        self.item_id_reverse_mapping = None
        self.user_id_reverse_mapping = None
        self.e_xy = None
        self.n_xy = None
        self.scores = None
        self.most_popular_items = None
        self.should_recommend_already_bought = False

    def initialize(self, **params):
        if 'should_recommend_already_bought' in params:
            self.should_recommend_already_bought = params['should_recommend_already_bought']

    def fit(self, interactions_df, users_df, items_df):
        """
        Training of the recommender.

        :param pd.DataFrame interactions_df: DataFrame with recorded interactions between users and items
            defined by user_id, item_id and features of the interaction.
        :param pd.DataFrame users_df: DataFrame with users and their features defined by
            user_id and the user feature columns.
        :param pd.DataFrame items_df: DataFrame with items and their features defined
            by item_id and the item feature columns.
        """

        # Shift item ids and user ids so that they are consecutive

        unique_item_ids = interactions_df['item_id'].unique()
        self.item_id_mapping = dict(zip(unique_item_ids, list(range(len(unique_item_ids)))))
        self.item_id_reverse_mapping = dict(zip(list(range(len(unique_item_ids))), unique_item_ids))
        unique_user_ids = interactions_df['user_id'].unique()
        self.user_id_mapping = dict(zip(unique_user_ids, list(range(len(unique_user_ids)))))
        self.user_id_reverse_mapping = dict(zip(list(range(len(unique_user_ids))), unique_user_ids))

        interactions_df = interactions_df.copy()
        interactions_df.replace({'item_id': self.item_id_mapping, 'user_id': self.user_id_mapping}, inplace=True)

        # Get the number of items and users

        self.interactions_df = interactions_df
        n_items = np.max(interactions_df['item_id']) + 1
        n_users = np.max(interactions_df['user_id']) + 1

        # Get maximal number of interactions

        n_user_interactions = interactions_df[['user_id', 'item_id']].groupby("user_id").count()
        # Unnecessary, but added for readability
        n_user_interactions = n_user_interactions.rename(columns={'item_id': 'n_items'})
        max_interactions = n_user_interactions['n_items'].max()

        # Calculate P_Y's

        n_interactions = len(interactions_df)
        p_y = interactions_df[['item_id', 'user_id']].groupby("item_id").count().reset_index()
        p_y = p_y.rename(columns={'user_id': 'P_Y'})
        p_y.loc[:, 'P_Y'] = p_y['P_Y'] / n_interactions
        p_y = dict(zip(p_y['item_id'], p_y['P_Y']))

        # Get the series of all items

        # items = list(range(n_items))
        items = interactions_df['item_id'].unique()

        # For every X calculate the E[Y|X]

        e_xy = np.zeros(shape=(n_items, n_items))
        e_xy[:][:] = -1e100

        p_y_powers = {}
        for y in items:
            p_y_powers[y] = np.array([p_y[y]**k for k in range(1, max_interactions + 1)])

        # In the next version calculate all alpha_k first (this works well with parallelization)

        for x in items:
            # Get users who bought X
            c_x = interactions_df.loc[interactions_df['item_id'] == x]['user_id'].unique()

            # Get users who bought only X
            c_only_x = interactions_df.loc[interactions_df['item_id'] != x]['user_id'].unique()
            c_only_x = list(set(c_x.tolist()) - set(c_only_x.tolist()))

            # Calculate the number of non-X interactions for each user who bought X
            # Include users with zero non-X interactions
            n_non_x_interactions = interactions_df.loc[interactions_df['item_id'] != x, ['user_id', 'item_id']]
            n_non_x_interactions = n_non_x_interactions.groupby("user_id").count()
            # Unnecessary, but added for readability
            n_non_x_interactions = n_non_x_interactions.rename(columns={'item_id': 'n_items'})

            zero_non_x_interactions = pd.DataFrame([[0]]*len(c_only_x), columns=["n_items"], index=c_only_x)  # Remove
            n_non_x_interactions = pd.concat([n_non_x_interactions, zero_non_x_interactions])

            n_non_x_interactions = n_non_x_interactions.loc[c_x.tolist()]

            # Calculate the expected numbers of Y products bought by clients who bought X
            alpha_k = np.array([np.sum([(-1)**(k + 1) * scisp.binom(abs_c, k)
                                        for abs_c in n_non_x_interactions["n_items"]])
                                for k in range(1, max_interactions + 1)])

            for y in items:  # Optimize to use only those Y's which have at least one client who bought both X and Y
                if y != x:
                    e_xy[x][y] = np.sum(alpha_k * p_y_powers[y])
                else:
                    e_xy[x][y] = n_users * p_y[x]

        self.e_xy = e_xy

        # Calculate the number of users who bought both X and Y

        # Simple and slow method (commented out)

        # n_xy = np.zeros(shape=(n_items, n_items))

        # for x in items:
        #     for y in items:
        #         users_x = set(interactions_df.loc[interactions_df['item_id'] == x]['user_id'].tolist())
        #         users_y = set(interactions_df.loc[interactions_df['item_id'] == y]['user_id'].tolist())
        #         users_x_and_y = users_x & users_y
        #         n_xy[x][y] = len(users_x_and_y)

        # Optimized method (can be further optimized by using sparse matrices)

        # Get the user-item interaction matrix (mapping to int is necessary because of how iterrows works)
        r = np.zeros(shape=(n_users, n_items))
        for idx, interaction in interactions_df.iterrows():
            r[int(interaction['user_id'])][int(interaction['item_id'])] = 1

        # Get the number of users who bought both X and Y

        n_xy = np.matmul(r.T, r)

        self.n_xy = n_xy

        self.scores = np.divide(n_xy - e_xy, np.sqrt(e_xy), out=np.zeros_like(n_xy), where=e_xy != 0)

        # Find the most popular items for the cold start problem

        offers_count = interactions_df.loc[:, ['item_id', 'user_id']].groupby(by='item_id').count()
        offers_count = offers_count.sort_values('user_id', ascending=False)
        self.most_popular_items = offers_count.index

    def recommend(self, users_df, items_df, n_recommendations=1):
        """
        Serving of recommendations. Scores items in items_df for each user in users_df and returns
        top n_recommendations for each user.

        :param pd.DataFrame users_df: DataFrame with users and their features for which
            recommendations should be generated.
        :param pd.DataFrame items_df: DataFrame with items and their features which should be scored.
        :param int n_recommendations: Number of recommendations to be returned for each user.
        :return: DataFrame with user_id, item_id and score as columns returning n_recommendations top recommendations
            for each user.
        :rtype: pd.DataFrame
        """

        # Clean previous recommendations (iloc could be used alternatively)
        self.recommender_df = self.recommender_df[:0]

        # Handle users not in the training data

        # Map item ids

        items_df = items_df.copy()
        items_df.replace({'item_id': self.item_id_mapping}, inplace=True)

        # Generate recommendations

        for idx, user in users_df.iterrows():
            recommendations = []

            user_id = user['user_id']

            if user_id in self.user_id_mapping:
                mapped_user_id = self.user_id_mapping[user_id]

                x_list = self.interactions_df.loc[self.interactions_df['user_id'] == mapped_user_id]['item_id'].tolist()
                final_scores = np.sum(self.scores[x_list], axis=0)

                # Choose n recommendations based on highest scores
                if not self.should_recommend_already_bought:
                    final_scores[x_list] = -1e100

                chosen_ids = np.argsort(-final_scores)[:n_recommendations]

                for item_id in chosen_ids:
                    recommendations.append(
                        {
                            'user_id': self.user_id_reverse_mapping[mapped_user_id],
                            'item_id': self.item_id_reverse_mapping[item_id],
                            'score': final_scores[item_id]
                        }
                    )
            else:  # For new users recommend most popular items
                for i in range(n_recommendations):
                    recommendations.append(
                        {
                            'user_id': user['user_id'],
                            'item_id': self.item_id_reverse_mapping[self.most_popular_items[i]],
                            'score': 1.0
                        }
                    )

            user_recommendations = pd.DataFrame(recommendations)

            self.recommender_df = pd.concat([self.recommender_df, user_recommendations])

        return self.recommender_df
