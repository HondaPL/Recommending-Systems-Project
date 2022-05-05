# Load libraries ---------------------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MultiLabelBinarizer

from recommenders.recommender import Recommender

# ------------------------------------------------------------


class LinearRegressionRecommender(Recommender):
    """
    Linear regression recommender class.
    """

    def __init__(self):
        """
        Initialize recommender params and variables.
        """
        super().__init__()
        self.model = None
        self.mlb = None
        self.users_dict = None
        self.user_features = None

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

        # Transform genres to a more code-friendly form

        interactions_df = pd.merge(interactions_df, items_df, on='item_id')
        interactions_df = self._transform_genres(interactions_df)

        # Prepare user features

        users_df = interactions_df[['user_id', 'genres']].copy()
        users_df = users_df.explode('genres')
        users_df['val'] = 1
        users_df = users_df.pivot_table(index='user_id', columns='genres', values='val', aggfunc='count')
        users_df = users_df / users_df.sum(axis=1).values.reshape(-1, 1)
        users_df = users_df.rename_axis(None, axis=1).fillna(0)
        users_df = users_df.add_prefix('user_')

        self.users_dict = users_df.to_dict('index')

        self.user_features = users_df.columns.tolist()

        interactions_df = interactions_df.merge(users_df, on='user_id')

        # Prepare item features

        # Transform genres into binary values

        self.mlb = MultiLabelBinarizer()
        interactions_df = interactions_df.join(
            pd.DataFrame(self.mlb.fit_transform(interactions_df.pop('genres')),
                         columns=self.mlb.classes_,
                         index=interactions_df.index))

        # Normalize the values so that each movie's genres sum up to 1

        interactions_df[self.mlb.classes_] = interactions_df[self.mlb.classes_] \
            / interactions_df[self.mlb.classes_].sum(axis=1).values.reshape(-1, 1)

        # Prepare input data and fit the model

        interactions_df[self.mlb.classes_] = interactions_df[self.mlb.classes_] \
            * interactions_df[self.user_features].values

        x = interactions_df.loc[:, list(self.mlb.classes_)].values
        y = interactions_df['rating'].values

        self.model = LinearRegression().fit(x, y)

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

        # Transform the item to be scored into proper features

        items_df = items_df.copy()
        items_df = self._transform_genres(items_df)

        items_df = items_df.join(
            pd.DataFrame(self.mlb.transform(items_df.pop('genres')),
                         columns=self.mlb.classes_,
                         index=items_df.index))

        items_df[self.mlb.classes_] = items_df[self.mlb.classes_] \
            / items_df[self.mlb.classes_].sum(axis=1).values.reshape(-1, 1)

        # Score the item

        recommendations = pd.DataFrame(columns=['user_id', 'item_id', 'score'])

        for ix, user in users_df.iterrows():
            if user['user_id'] in self.users_dict:
                user_df = pd.DataFrame.from_dict({user['user_id']: self.users_dict[user['user_id']]}, orient='index')
            else:
                user_df = pd.DataFrame.from_dict(
                    {user['user_id']: [1 / len(self.user_features)]*len(self.user_features)}, orient='index')
#             display(user_df)
#             display(items_df)
            input_df = items_df.copy()
            input_df[self.mlb.classes_] = items_df[self.mlb.classes_] * user_df.values
#             display(input_df)
            scores = self.model.predict(input_df.loc[:, self.mlb.classes_].values)

            chosen_pos = np.argsort(-scores)[:n_recommendations]

            user_recommendations = []
            for item_pos in chosen_pos:
                user_recommendations.append(
                    {
                        'user_id': user['user_id'],
                        'item_id': input_df.iloc[item_pos]['item_id'],
                        'score': scores[item_pos]
                    }
                )

            user_recommendations = pd.DataFrame(user_recommendations)

            recommendations = pd.concat([recommendations, user_recommendations])

        return recommendations

    @staticmethod
    def _transform_genres(df):
        """
        Transforms a string with genres into a list of cleaned genre names.

        :param pd.DataFrame df: A DataFrame with 'genres' column.
        """
        df.loc[:, 'genres'] = df['genres'].str.replace("-", "_", regex=False)
        df.loc[:, 'genres'] = df['genres'].str.replace(" ", "_", regex=False)
        df.loc[:, 'genres'] = df['genres'].str.replace("(", "", regex=False)
        df.loc[:, 'genres'] = df['genres'].str.replace(")", "", regex=False)
        df.loc[:, 'genres'] = df['genres'].str.lower()
        df.loc[:, 'genres'] = df['genres'].str.split("|")
        return df


class SVRRecommender(Recommender):
    """
    SVR recommender class.
    """

    def __init__(self, kernel='rbf', c=1.0, epsilon=0.1):
        """
        Initialize base recommender params and variables.
        """
        super().__init__()
        self.model = None
        self.mlb = None
        self.kernel = kernel
        self.c = c
        self.epsilon = epsilon
        self.mlb = None
        self.users_dict = None
        self.user_features = None

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

        # Transform genres to a more code-friendly form

        interactions_df = pd.merge(interactions_df, items_df, on='item_id')
        interactions_df = self._transform_genres(interactions_df)

        # Prepare user features

        users_df = interactions_df[['user_id', 'genres']].copy()
        users_df = users_df.explode('genres')
        users_df['val'] = 1
        users_df = users_df.pivot_table(index='user_id', columns='genres', values='val', aggfunc='count')
        users_df = users_df / users_df.sum(axis=1).values.reshape(-1, 1)
        users_df = users_df.rename_axis(None, axis=1).fillna(0)
        users_df = users_df.add_prefix('user_')
#         display(users_df.head(10))

        self.users_dict = users_df.to_dict('index')

        self.user_features = users_df.columns.tolist()

        interactions_df = interactions_df.merge(users_df, on='user_id')
#         display(interactions_df.head(10))

        # Prepare item features

        # Transform genres into binary values

        self.mlb = MultiLabelBinarizer()
        interactions_df = interactions_df.join(
            pd.DataFrame(self.mlb.fit_transform(interactions_df.pop('genres')),
                         columns=self.mlb.classes_,
                         index=interactions_df.index))

        # Normalize the values so that each movie's genres sum up to 1

        interactions_df[self.mlb.classes_] = interactions_df[self.mlb.classes_] \
            / interactions_df[self.mlb.classes_].sum(axis=1).values.reshape(-1, 1)

#         display(interactions_df.loc[:, self.mlb.classes_].head(10))

        # Prepare input data and fit the model

        interactions_df[self.mlb.classes_] = interactions_df[self.mlb.classes_] \
            * interactions_df[self.user_features].values

#         display(interactions_df.head(10))

        x = interactions_df.loc[:, list(self.mlb.classes_)].values
        y = interactions_df['rating'].values

        self.model = SVR(kernel=self.kernel, C=self.c, epsilon=self.epsilon).fit(x, y)

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

        # Transform the item to be scored into proper features

        items_df = items_df.copy()
        items_df = self._transform_genres(items_df)

        items_df = items_df.join(
            pd.DataFrame(self.mlb.transform(items_df.pop('genres')),
                         columns=self.mlb.classes_,
                         index=items_df.index))

        items_df[self.mlb.classes_] = items_df[self.mlb.classes_] \
            / items_df[self.mlb.classes_].sum(axis=1).values.reshape(-1, 1)

        # Score the item

        recommendations = pd.DataFrame(columns=['user_id', 'item_id', 'score'])

        for ix, user in users_df.iterrows():
            if user['user_id'] in self.users_dict:
                user_df = pd.DataFrame.from_dict({user['user_id']: self.users_dict[user['user_id']]}, orient='index')
            else:
                user_df = pd.DataFrame.from_dict(
                    {user['user_id']: [1 / len(self.user_features)]*len(self.user_features)}, orient='index')
#             display(user_df)
#             display(items_df)
            input_df = items_df.copy()
            input_df[self.mlb.classes_] = items_df[self.mlb.classes_] * user_df.values
#             display(input_df)
            scores = self.model.predict(input_df.loc[:, self.mlb.classes_].values)

            chosen_pos = np.argsort(-scores)[:n_recommendations]

            user_recommendations = []
            for item_pos in chosen_pos:
                user_recommendations.append(
                    {
                        'user_id': user['user_id'],
                        'item_id': input_df.iloc[item_pos]['item_id'],
                        'score': scores[item_pos]
                    }
                )

            user_recommendations = pd.DataFrame(user_recommendations)

            recommendations = pd.concat([recommendations, user_recommendations])

        return recommendations

    @staticmethod
    def _transform_genres(df):
        """
        Transforms a string with genres into a list of cleaned genre names.

        :param pd.DataFrame df: A DataFrame with 'genres' column.
        """
        df.loc[:, 'genres'] = df['genres'].str.replace("-", "_", regex=False)
        df.loc[:, 'genres'] = df['genres'].str.replace(" ", "_", regex=False)
        df.loc[:, 'genres'] = df['genres'].str.replace("(", "", regex=False)
        df.loc[:, 'genres'] = df['genres'].str.replace(")", "", regex=False)
        df.loc[:, 'genres'] = df['genres'].str.lower()
        df.loc[:, 'genres'] = df['genres'].str.split("|")
        return df
