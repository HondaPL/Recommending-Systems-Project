# Load libraries ---------------------------------------------

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from livelossplot import PlotLosses
from collections import deque

from recommenders.recommender import Recommender

# ------------------------------------------------------------


class GMFModel(nn.Module):
    def __init__(self, n_items, n_users, embedding_dim, seed):
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, x):
        user_ids = x[:, 0]
        item_ids = x[:, 1]
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        x = self.fc(user_embedding * item_embedding)
        x = torch.sigmoid(x)

        return x


class GMFRecommender(Recommender):
    """
    General Matrix Factorization recommender as described in:
    - He X., Liao L., Zhang H., Nie L., Hu X., Chua T., Neural Collaborative Filtering, WWW Conference, 2017
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

        self.nn_model = None
        self.optimizer = None

        self.n_neg_per_pos = n_neg_per_pos
        if 'n_epochs' in params:  # number of epochs (each epoch goes through the entire training set)
            self.n_epochs = params['n_epochs']
        else:
            self.n_epochs = 10
        if 'lr' in params:  # learning rate
            self.lr = params['lr']
        else:
            self.lr = 0.01
        if 'weight_decay' in params:  # weight decay (L2 regularization)
            self.weight_decay = params['weight_decay']
        else:
            self.weight_decay = 0.001
        if 'embedding_dim' in params:
            self.embedding_dim = params['embedding_dim']
        else:
            self.embedding_dim = 4
        if 'batch_size' in params:
            self.batch_size = params['batch_size']
        else:
            self.batch_size = 64
        if 'device' in params:
            self.device = params['device']
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if 'should_recommend_already_bought' in params:
            self.should_recommend_already_bought = params['should_recommend_already_bought']
        else:
            self.should_recommend_already_bought = False

        if 'train' in params:
            self.train = params['train']
        else:
            self.train = False
        self.validation_set_size = 0.2

        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)
        torch.manual_seed(seed)

        if 'should_save_model' in params:
            self.should_save_model = params['should_save_model']
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

        self.interactions_df = interactions_df.copy()
        n_users = np.max(interactions_df['user_id']) + 1
        n_items = np.max(interactions_df['item_id']) + 1

        # Get the user-item interaction matrix (mapping to int is necessary because of how iterrows works)
        r = np.zeros(shape=(n_users, n_items))
        for idx, interaction in interactions_df.iterrows():
            r[int(interaction['user_id'])][int(interaction['item_id'])] = 1

        self.r = r

        # Indicate positive interactions

        interactions_df.loc[:, 'interacted'] = 1

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
        interactions_df = interactions_df.reset_index(drop=True)

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

        # Initialize the network

        self.nn_model = GMFModel(n_items, n_users, self.embedding_dim, self.seed)
        self.nn_model.train()
        self.nn_model.to(self.device)
        self.optimizer = optim.Adam(self.nn_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Split the data

        if self.train:
            interaction_ids = self.rng.permutation(len(interactions_df))
            train_validation_slice_idx = int(len(interactions_df) * (1 - self.validation_set_size))
            training_ids = interaction_ids[:train_validation_slice_idx]
            validation_ids = interaction_ids[train_validation_slice_idx:]
        else:
            interaction_ids = self.rng.permutation(len(interactions_df))
            training_ids = interaction_ids
            validation_ids = []

        # Train the model

        for epoch in range(self.n_epochs):
            if self.print_type is not None and self.print_type == 'live':
                logs = {}

            # Train

            training_losses.clear()
            training_total_loss = 0.0

            self.rng.shuffle(training_ids)

            n_batches = int(np.ceil(len(training_ids) / self.batch_size))

            for batch_idx in range(n_batches):

                batch_ids = training_ids[(batch_idx * self.batch_size):((batch_idx + 1) * self.batch_size)]

                batch = interactions_df.loc[batch_ids]
                batch_input = torch.from_numpy(batch.loc[:, ['user_id', 'item_id']].values).long().to(self.device)
                y_target = torch.from_numpy(batch.loc[:, ['interacted']].values).float().to(self.device)

                # Create responses

                y = self.nn_model(batch_input).clip(0.000001, 0.999999)

                # Define loss and backpropagate

                self.optimizer.zero_grad()
                loss = -(y_target * y.log() + (1 - y_target) * (1 - y).log()).sum()

                loss.backward()
                self.optimizer.step()

                training_total_loss += loss.item()

                if self.print_type is not None and self.print_type == 'text':
                    print("\rEpoch: {}\tBatch: {}\tLast epoch - avg training loss: {:.2f} avg validation loss: {:.2f} loss: {}".format(
                        epoch, batch_idx, last_training_total_loss, last_validation_total_loss, loss), end="")

                training_losses.append(loss.item())
                training_avg_losses.append(np.mean(training_losses))

            # Validate

            validation_total_loss = 0.0

            batch = interactions_df.loc[validation_ids]
            batch_input = torch.from_numpy(batch.loc[:, ['user_id', 'item_id']].values).long().to(self.device)
            y_target = torch.from_numpy(batch.loc[:, ['interacted']].values).float().to(self.device)

            # Create responses

            y = self.nn_model(batch_input).clip(0.000001, 0.999999)

            # Calculate validation loss

            loss = -(y_target * y.log() + (1 - y_target) * (1 - y).log()).sum()
            validation_total_loss += loss.item()

            # Save and print epoch losses

            training_last_avg_loss = training_total_loss / len(training_ids)
            if self.train:
                validation_last_avg_loss = validation_total_loss / len(validation_ids)

            if self.print_type is not None and self.print_type == 'live' and epoch >= 0:
                # A bound on epoch prevents showing extremely high losses in the first epochs
                logs['loss'] = training_last_avg_loss
                if self.train:
                    logs['val_loss'] = validation_last_avg_loss
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

                net_input = torch.tensor(list(zip([mapped_user_id]*len(ids_list), ids_list))).to(self.device)

                scores = self.nn_model(net_input).flatten().detach().cpu().numpy()

                # Choose n recommendations based on highest scores
                if not self.should_recommend_already_bought:
                    x_list = self.interactions_df.loc[
                        self.interactions_df['user_id'] == mapped_user_id]['item_id'].tolist()
                    scores[id_to_pos[x_list]] = -np.inf

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
        return self.nn_model.user_embedding(torch.tensor(mapped_user_id).to(self.device)).detach().cpu().numpy()

    def get_item_repr(self, item_id):
        mapped_item_id = self.item_id_mapping[item_id]
        return self.nn_model.item_embedding(torch.tensor(mapped_item_id).to(self.device)).detach().cpu().numpy()


class MLPModel(nn.Module):
    def __init__(self, n_items, n_users, embedding_dim, seed):
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.fc1 = nn.Linear(2 * embedding_dim, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)
        self.fc3 = nn.Linear(16, 1, bias=False)

    def forward(self, x):
        user = x[:, 0]
        item = x[:, 1]
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


class NeuMFModel(nn.Module):
    def __init__(self, n_items, n_users, gmf_embedding_dim, mlp_embedding_dim, seed):
        super().__init__()

        self.seed = torch.manual_seed(seed)

        # GMF

        self.gmf_user_embedding = nn.Embedding(n_users, gmf_embedding_dim)
        self.gmf_item_embedding = nn.Embedding(n_items, gmf_embedding_dim)

        # MLP

        self.mlp_user_embedding = nn.Embedding(n_users, mlp_embedding_dim)
        self.mlp_item_embedding = nn.Embedding(n_items, mlp_embedding_dim)
        self.mlp_fc1 = nn.Linear(2 * mlp_embedding_dim, 32, bias=False)
        self.mlp_fc2 = nn.Linear(32, 16, bias=False)

        # Merge

        self.fc = nn.Linear(32, 1, bias=False)

    def forward(self, x):
        user = x[:, 0]
        item = x[:, 1]

        # GMF

        gmf_user_embedding = self.gmf_user_embedding(user)
        gmf_item_embedding = self.gmf_item_embedding(item)
        gmf_x = gmf_user_embedding * gmf_item_embedding

        # MLP

        mlp_user_embedding = self.mlp_user_embedding(user)
        mlp_item_embedding = self.mlp_item_embedding(item)
        mlp_x = torch.cat([mlp_user_embedding, mlp_item_embedding], dim=1)
        mlp_x = torch.relu(self.mlp_fc1(mlp_x))
        mlp_x = torch.relu(self.mlp_fc2(mlp_x))

        # Final score

        x = torch.cat([gmf_x, mlp_x], dim=1)
        x = torch.sigmoid(self.fc(x))

        return x
