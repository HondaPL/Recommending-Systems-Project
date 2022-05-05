# Load libraries ---------------------------------------------

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

from recommenders.recommender import Recommender

# ------------------------------------------------------------


class TFIDFRecommender(Recommender):
    """
    Recommender based on the TF-IDF method.
    """

    def __init__(self):
        """
        Initialize base recommender params and variables.
        """
        super().__init__()
        self.tfidf_scores = None

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

        self.tfidf_scores = defaultdict(lambda: 0.0)

        # Prepare the corpus for tfidf calculation

        interactions_df = pd.merge(interactions_df, items_df, on='item_id')
        user_genres = interactions_df.loc[:, ['user_id', 'genres']]
        user_genres.loc[:, 'genres'] = user_genres['genres'].str.replace("-", "_", regex=False)
        user_genres.loc[:, 'genres'] = user_genres['genres'].str.replace(" ", "_", regex=False)
        user_genres = user_genres.groupby('user_id').aggregate(lambda x: "|".join(x))
        user_genres.loc[:, 'genres'] = user_genres['genres'].str.replace("|", " ", regex=False)
        user_ids = user_genres.index.tolist()
        genres_corpus = user_genres['genres'].tolist()

        # Calculate tf-idf scores

        vectorizer = TfidfVectorizer()
        tfidf_scores = vectorizer.fit_transform(genres_corpus)

        # Transform results into a dict {(user_id, genre): score}

        for u in range(tfidf_scores.shape[0]):
            for g in range(tfidf_scores.shape[1]):
                self.tfidf_scores[(user_ids[u], vectorizer.get_feature_names()[g])] = tfidf_scores[u, g]

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

        # Transform genres to a unified form used by the vectorizer

        items_df = items_df.copy()
        items_df.loc[:, 'genres'] = items_df['genres'].str.replace("-", "_", regex=False)
        items_df.loc[:, 'genres'] = items_df['genres'].str.replace(" ", "_", regex=False)
        items_df.loc[:, 'genres'] = items_df['genres'].str.lower()
        items_df.loc[:, 'genres'] = items_df['genres'].str.split("|")

        # Score items

        for uix, user in users_df.iterrows():
            items = []
            for iix, item in items_df.iterrows():
                score = 0.0
                for genre in item['genres']:
                    score += self.tfidf_scores[(user['user_id'], genre)]
                score /= len(item['genres'])
                items.append((item['item_id'], score))

            items = sorted(items, key=lambda x: x[1], reverse=True)
            user_recommendations = pd.DataFrame({'user_id': user['user_id'],
                                                 'item_id': [item[0] for item in items][:n_recommendations],
                                                 'score': [item[1] for item in items][:n_recommendations]})

            recommendations = pd.concat([recommendations, user_recommendations])

        return recommendations
