# Load libraries ---------------------------------------------

import pandas as pd
import numpy as np

from recommenders.recommender import Recommender

# ------------------------------------------------------------


class NearestNeighborsRecommender(Recommender):
    """
    Nearest neighbors recommender allowing to do user-based or item-based collaborative filtering.

    Possible similarity measures:
        - 'cosine',
        - 'pearson'.
    """

    def __init__(self, **params):
        super().__init__()
        self.recommender_df = pd.DataFrame(columns=['user_id', 'item_id', 'score'])
        self.interactions_df = None
        self.item_id_mapping = None
        self.user_id_mapping = None
        self.item_id_reverse_mapping = None
        self.user_id_reverse_mapping = None
        self.r = None
        self.similarities = None
        self.most_popular_items = None

        self.collaboration_type = 'user'
        self.similarity_measure = 'cosine'
        self.n_neighbors = 10
        self.should_recommend_already_bought = False

        if 'n_neighbors' in params:
            self.n_neighbors = params['n_neighbors']
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

        del users_df, items_df

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

        # Get the user-item interaction matrix (mapping to int is necessary because of how iterrows works)
        r = np.zeros(shape=(n_users, n_items))
        for idx, interaction in interactions_df.iterrows():
            r[int(interaction['user_id'])][int(interaction['item_id'])] = 1

        if self.collaboration_type == 'item':
            r = r.T

        self.r = r

        # Calculate all similarities

        similarities = None
        if self.similarity_measure == 'cosine':
            n_uv = np.matmul(r, r.T)
            norms = np.sqrt(np.diag(n_uv))
            similarities = n_uv / norms[:, np.newaxis] / norms[np.newaxis, :]
        elif self.similarity_measure == 'pearson':
            r_shifted = r - np.mean(r, axis=1).reshape(-1, 1)
            n_uv = np.matmul(r_shifted, r_shifted.T)
            norms = np.sqrt(np.diag(n_uv))
            norms[norms == 0] = 0.000001
            similarities = n_uv / norms[:, np.newaxis] / norms[np.newaxis, :]

        np.fill_diagonal(similarities, -1000)

        self.similarities = similarities

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
        items_df = items_df.loc[items_df['item_id'].isin(self.item_id_mapping)]
        items_df.replace({'item_id': self.item_id_mapping}, inplace=True)

        # Generate recommendations

        for idx, user in users_df.iterrows():
            recommendations = []

            user_id = user['user_id']

            if user_id in self.user_id_mapping:
                chosen_ids = []
                scores = []
                mapped_user_id = self.user_id_mapping[user_id]

                if self.collaboration_type == 'user':
                    neighbor_ids = np.argsort(-self.similarities[mapped_user_id])[:self.n_neighbors]
                    user_similarities = self.similarities[mapped_user_id][neighbor_ids]

                    item_ids = items_df['item_id'].tolist()

                    v_i = self.r[neighbor_ids][:, item_ids]

                    scores = np.matmul(user_similarities, v_i) / np.sum(user_similarities)

                    # Choose n recommendations based on highest scores
                    if not self.should_recommend_already_bought:
                        x_list = self.interactions_df.loc[
                            self.interactions_df['user_id'] == mapped_user_id]['item_id'].tolist()
                        scores[x_list] = -1e100

                    chosen_ids = np.argsort(-scores)[:n_recommendations]

                elif self.collaboration_type == 'item':
                    x_list = self.interactions_df.loc[
                        self.interactions_df['user_id'] == mapped_user_id]['item_id'].tolist()
                    scores = np.sum(self.similarities[x_list], axis=0)

                    # Choose n recommendations based on highest scores
                    if not self.should_recommend_already_bought:
                        scores[x_list] = -1e100

                    chosen_ids = np.argsort(-scores)[:n_recommendations]

                for item_id in chosen_ids:
                    recommendations.append(
                        {
                            'user_id': self.user_id_reverse_mapping[mapped_user_id],
                            'item_id': self.item_id_reverse_mapping[item_id],
                            'score': scores[item_id]
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


class UserBasedCosineNearestNeighborsRecommender(NearestNeighborsRecommender):

    def __init__(self, **params):
        super().__init__(**params)

        self.collaboration_type = 'user'
        self.similarity_measure = 'cosine'


class UserBasedPearsonNearestNeighborsRecommender(NearestNeighborsRecommender):

    def __init__(self, **params):
        super().__init__(**params)

        self.collaboration_type = 'user'
        self.similarity_measure = 'pearson'


class ItemBasedCosineNearestNeighborsRecommender(NearestNeighborsRecommender):

    def __init__(self, **params):
        super().__init__(**params)

        self.collaboration_type = 'item'
        self.similarity_measure = 'cosine'


class ItemBasedPearsonNearestNeighborsRecommender(NearestNeighborsRecommender):

    def __init__(self, **params):
        super().__init__(**params)

        self.collaboration_type = 'item'
        self.similarity_measure = 'pearson'
