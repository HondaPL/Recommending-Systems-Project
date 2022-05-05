# Load libraries ---------------------------------------------

import pandas as pd
import numpy as np

from recommenders.recommender import Recommender

# ------------------------------------------------------------


class RandomRecommender(Recommender):
    """
    Base recommender class.
    """

    def __init__(self, seed=0):
        """
        Initialize base recommender params and variables.

        :param int seed: Seed for the random number generator.
        """
        super().__init__()
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)
        self.items = []

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
        self.items = items_df['item_id'].unique().tolist()

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

        recommendations = pd.DataFrame(columns=['user_id', 'item_id', 'score'])

        for ix, user in users_df.iterrows():
            user_recommendations = pd.DataFrame({'user_id': user['user_id'],
                                                 'item_id': self.rng.choice(self.items, n_recommendations,
                                                                            replace=False),
                                                 'score': 1.0})

            recommendations = pd.concat([recommendations, user_recommendations])

        recommendations = recommendations.reset_index(drop=True)

        return recommendations


class MostPopularRecommender(Recommender):
    """
    Most popular recommender class.
    """

    def __init__(self):
        """
        Initialize recommender params and variables.
        """
        super().__init__()
        self.most_popular_items = None

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
        offers_count = interactions_df.loc[:, ['item_id', 'user_id']].groupby(by='item_id').count()
        self.most_popular_items = offers_count.sort_values('user_id', ascending=False).rename(
            columns={'user_id': 'popularity'})

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

        recommendations = pd.DataFrame(columns=['user_id', 'item_id', 'score'])

        for ix, user in users_df.iterrows():
            user_recommendations = pd.DataFrame({'user_id': [user['user_id']]*n_recommendations,
                                                 'item_id': self.most_popular_items.index[:n_recommendations],
                                                 'score': self.most_popular_items.popularity[:n_recommendations]})

            recommendations = pd.concat([recommendations, user_recommendations])

        return recommendations


class HighestRatedRecommender(Recommender):
    """
    Highest rated recommender class.
    """

    def __init__(self):
        """
        Initialize recommender params and variables.
        """
        super().__init__()
        self.offer_ratings = None

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
        offer_ratings = interactions_df.loc[:, ['item_id', 'rating']].groupby(by='item_id').mean()
        offer_counts = interactions_df.loc[:, ['item_id', 'rating']].groupby(by='item_id').count()
        offer_ratings = offer_ratings.loc[offer_counts['rating'] >= 50]
        self.offer_ratings = offer_ratings.sort_values('rating', ascending=False)

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

        recommendations = pd.DataFrame(columns=['user_id', 'item_id', 'score'])

        for ix, user in users_df.iterrows():
            user_recommendations = pd.DataFrame({'user_id': [user['user_id']]*n_recommendations,
                                                 'item_id': self.offer_ratings.index[:n_recommendations],
                                                 'score': self.offer_ratings.rating[:n_recommendations]})

            recommendations = pd.concat([recommendations, user_recommendations])

        return recommendations

class SVRRecommender(Recommender):
    """
    SVR recommender class.
    """
    
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        """
        Initialize base recommender params and variables.
        """
        self.model = None
        self.mlb = None
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
    
    def fit(self, interactions_df, users_df, items_df):
        """
        Training of the recommender.
        
        :param pd.DataFrame interactions_df: DataFrame with recorded interactions between users and items 
            defined by user_id, item_id and features of the interaction.
        :param pd.DataFrame users_df: DataFrame with users and their features defined by user_id and the user feature columns.
        :param pd.DataFrame items_df: DataFrame with items and their features defined by item_id and the item feature columns.
        """
        
        interactions_df = pd.merge(interactions_df, items_df, on='item_id')
        interactions_df.loc[:, 'genres'] = interactions_df['genres'].str.replace("-", "_", regex=False)
        interactions_df.loc[:, 'genres'] = interactions_df['genres'].str.replace(" ", "_", regex=False)
        interactions_df.loc[:, 'genres'] = interactions_df['genres'].str.lower()
        interactions_df.loc[:, 'genres'] = interactions_df['genres'].str.split("|")
        
        self.mlb = MultiLabelBinarizer()
        interactions_df = interactions_df.join(
            pd.DataFrame(self.mlb.fit_transform(interactions_df.pop('genres')),
                         columns=self.mlb.classes_,
                         index=interactions_df.index))
        
#         print(interactions_df.head())
        
        x = interactions_df.loc[:, self.mlb.classes_].values
        y = interactions_df['rating'].values
    
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon).fit(x, y)
    
    def recommend(self, users_df, items_df, n_recommendations=1):
        """
        Serving of recommendations. Scores items in items_df for each user in users_df and returns 
        top n_recommendations for each user.
        
        :param pd.DataFrame users_df: DataFrame with users and their features for which recommendations should be generated.
        :param pd.DataFrame items_df: DataFrame with items and their features which should be scored.
        :param int n_recommendations: Number of recommendations to be returned for each user.
        :return: DataFrame with user_id, item_id and score as columns returning n_recommendations top recommendations 
            for each user.
        :rtype: pd.DataFrame
        """
        
        # Transform the item to be scored into proper features
        
        items_df = items_df.copy()
        items_df.loc[:, 'genres'] = items_df['genres'].str.replace("-", "_", regex=False)
        items_df.loc[:, 'genres'] = items_df['genres'].str.replace(" ", "_", regex=False)
        items_df.loc[:, 'genres'] = items_df['genres'].str.lower()
        items_df.loc[:, 'genres'] = items_df['genres'].str.split("|")
        
        items_df = items_df.join(
            pd.DataFrame(self.mlb.transform(items_df.pop('genres')),
                         columns=self.mlb.classes_,
                         index=items_df.index))
        
#         print(items_df)
        
        # Score the item
    
        recommendations = pd.DataFrame(columns=['user_id', 'item_id', 'score'])
        
        for ix, user in users_df.iterrows():
            score = self.model.predict(items_df.loc[:, self.mlb.classes_].values)[0]
                
            user_recommendations = pd.DataFrame({'user_id': [user['user_id']],
                                                 'item_id': items_df.iloc[0]['item_id'],
                                                 'score': score})

            recommendations = pd.concat([recommendations, user_recommendations])

        return recommendations
        
svr_recommender = SVRRecommender(C=5.9672721141155, epsilon=0.8583733904374324)