# Load libraries ---------------------------------------------

import pandas as pd
import numpy as np
import scipy.special as scisp
from livelossplot import PlotLosses
from collections import defaultdict, deque

from recommenders.recommender import Recommender

# ------------------------------------------------------------


class NetflixRecommender(Recommender):
    """
    Collaborative filtering based on matrix factorization with the following choice of an optimizer:
      - Stochastic Gradient Descent (SGD),
      - Mini-Batch Gradient Descent (MBGD),
      - Alternating Least Squares (ALS).
    """

    def __init__(self, seed=6789, n_neg_per_pos=5, print_type=None, **params):
        super().__init__()
        self.recommender_df = pd.DataFrame(columns=['user_id', 'item_id', 'score'])
        self.interactions_df = None
        self.item_id_mapping = None
        self.user_id_mapping = None
        self.item_id_reverse_mapping = None
        self.user_id_reverse_mapping = None
        self.r = None
        self.most_popular_items = None

        self.n_neg_per_pos = n_neg_per_pos
        if 'optimizer' in params:
            self.optimizer = params['optimizer']
        else:
            self.optimizer = 'SGD'
        if 'n_epochs' in params:  # number of epochs (each epoch goes through the entire training set)
            self.n_epochs = params['n_epochs']
        else:
            self.n_epochs = 10
        if 'lr' in params:  # learning rate
            self.lr = params['lr']
        else:
            self.lr = 0.01
        if 'reg_l' in params:  # regularization coefficient
            self.reg_l = params['reg_l']
        else:
            self.reg_l = 0.1
        if 'embedding_dim' in params:
            self.embedding_dim = params['embedding_dim']
        else:
            self.embedding_dim = 8

        self.user_repr = None
        self.item_repr = None

        if 'should_recommend_already_bought' in params:
            self.should_recommend_already_bought = params['should_recommend_already_bought']
        else:
            self.should_recommend_already_bought = False

        self.validation_set_size = 0.2

        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

        self.print_type = print_type

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
        n_users = np.max(interactions_df['user_id']) + 1
        n_items = np.max(interactions_df['item_id']) + 1

        # Get the user-item interaction matrix (mapping to int is necessary because of how iterrows works)
        r = np.zeros(shape=(n_users, n_items))
        for idx, interaction in interactions_df.iterrows():
            r[int(interaction['user_id'])][int(interaction['item_id'])] = 1

        self.r = r

        # Generate negative interactions
        negative_interactions = []

        i = 0
        while i < self.n_neg_per_pos * len(interactions_df):
            sample_size = 1000
            user_ids = self.rng.choice(np.arange(n_users), size=sample_size)
            item_ids = self.rng.choice(np.arange(n_items), size=sample_size)

            j = 0
            while j < sample_size and i < self.n_neg_per_pos * len(interactions_df):
                if r[user_ids[j]][item_ids[j]] == 0:
                    negative_interactions.append([user_ids[j], item_ids[j], 0])
                    i += 1
                j += 1

        interactions_df = pd.concat(
            [interactions_df, pd.DataFrame(negative_interactions, columns=['user_id', 'item_id', 'interacted'])])

        # Initialize user and item embeddings as random vectors (from Gaussian distribution)

        self.user_repr = self.rng.normal(0, 1, size=(r.shape[0], self.embedding_dim))
        self.item_repr = self.rng.normal(0, 1, size=(r.shape[1], self.embedding_dim))

        # Initialize losses and loss visualization

        if self.print_type is not None and self.print_type == 'live':
            liveloss = PlotLosses()

        training_losses = deque(maxlen=50)
        training_avg_losses = []
        training_epoch_losses = []
        validation_losses = deque(maxlen=50)
        validation_avg_losses = []
        validation_epoch_losses = []
        last_training_total_loss = 0.0
        last_validation_total_loss = 0.0

        # Split the data

        interaction_ids = self.rng.permutation(len(interactions_df))
        train_validation_slice_idx = int(len(interactions_df) * (1 - self.validation_set_size))
        training_ids = interaction_ids[:train_validation_slice_idx]
        validation_ids = interaction_ids[train_validation_slice_idx:]

        # Train the model

        for epoch in range(self.n_epochs):
            if self.print_type is not None and self.print_type == 'live':
                logs = {}

            # Train

            training_losses.clear()
            training_total_loss = 0.0
            batch_idx = 0
            for idx in training_ids:
                user_id = int(interactions_df.iloc[idx]['user_id'])
                item_id = int(interactions_df.iloc[idx]['item_id'])

                e_ui = r[user_id, item_id] - np.dot(self.user_repr[user_id], self.item_repr[item_id])
                self.user_repr[user_id] = self.user_repr[user_id] \
                    + self.lr * (e_ui * self.item_repr[item_id] - self.reg_l * self.user_repr[user_id])
                self.item_repr[item_id] = self.item_repr[item_id] \
                    + self.lr * (e_ui * self.user_repr[user_id] - self.reg_l * self.item_repr[item_id])

                loss = e_ui**2
                training_total_loss += loss

                if self.print_type is not None and self.print_type == 'text':
                    print("\rEpoch: {}\tBatch: {}\tLast epoch - avg training loss: {:.2f} avg validation loss: {:.2f} loss: {}".format(
                        epoch, batch_idx, last_training_total_loss, last_validation_total_loss, loss), end="")

                batch_idx += 1

                training_losses.append(loss)
                training_avg_losses.append(np.mean(training_losses))

            # Validate

            validation_losses.clear()
            validation_total_loss = 0.0
            for idx in validation_ids:
                user_id = int(interactions_df.iloc[idx]['user_id'])
                item_id = int(interactions_df.iloc[idx]['item_id'])

                e_ui = r[user_id, item_id] - np.dot(self.user_repr[user_id], self.item_repr[item_id])

                loss = e_ui**2
                validation_total_loss += loss

                validation_losses.append(loss)
                validation_avg_losses.append(np.mean(validation_losses))

            # Save and print epoch losses

            training_last_avg_loss = training_total_loss / len(training_ids)
            training_epoch_losses.append(training_last_avg_loss)
            validation_last_avg_loss = validation_total_loss / len(validation_ids)
            validation_epoch_losses.append(validation_last_avg_loss)

            if self.print_type is not None and self.print_type == 'live' and epoch >= 3:
                # A bound on epoch prevents showing extremely high losses in the first epochs
                # noinspection PyUnboundLocalVariable
                logs['loss'] = training_last_avg_loss
                logs['val_loss'] = validation_last_avg_loss
                # noinspection PyUnboundLocalVariable
                liveloss.update(logs)
                liveloss.send()

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
                mapped_user_id = self.user_id_mapping[user_id]

                ids_list = items_df['item_id'].tolist()
                id_to_pos = np.array([0]*len(ids_list))
                for k in range(len(ids_list)):
                    id_to_pos[ids_list[k]] = k
                scores = np.matmul(self.user_repr[mapped_user_id].reshape(1, -1),
                                   self.item_repr[ids_list].T).flatten()

                # Choose n recommendations based on highest scores
                if not self.should_recommend_already_bought:
                    x_list = self.interactions_df.loc[
                        self.interactions_df['user_id'] == mapped_user_id]['item_id'].tolist()
                    scores[id_to_pos[x_list]] = -1e100

                chosen_pos = np.argsort(-scores)[:n_recommendations]

                for item_pos in chosen_pos:
                    recommendations.append(
                        {
                            'user_id': self.user_id_reverse_mapping[mapped_user_id],
                            'item_id': self.item_id_reverse_mapping[ids_list[item_pos]],
                            'score': scores[item_pos]
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

    def get_user_repr(self, user_id):
        mapped_user_id = self.user_id_mapping[user_id]
        return self.user_repr[mapped_user_id]

    def get_item_repr(self, item_id):
        mapped_item_id = self.item_id_mapping[item_id]
        return self.item_repr[mapped_item_id]
