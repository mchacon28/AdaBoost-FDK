# -*- coding: utf-8 -*-
"""
Implementation of AdaBoost for local models (i.e. building models at each client side without communication)

Authors:
    Mario Chacon-Falcon <mario.chacon.falcon@gmail.com>
    Jose M. Moyano <jmoyano1@us.es>

Created on:
    2024-06-23

Version:
    1.0.1

License:
    GNU General Public License v3
"""

import math

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder


class LocalAdaBoost():
    """
    Class that implements the base AdaBoost algorithm.
    It is used to train local models at each clients' side, without performing the federated process.
    """

    def __init__(self, n_estimators, classifier=DecisionTreeClassifier, params={}, random_state=0):
        """
        Args:
            n_estimators (int): number of classifiers in the ensemble
            classifier (class): The learning algorithm (from sklearn) used to train the base models.
            params (dict): Parameters for the classifier class
            random_state: Random seed for reproducibility of results, if neccessary.
        """
        self.n_estimators = n_estimators
        self.classifier = classifier
        self.params = params
        self.random_state = random_state

        self.model_weights = None
        self.models_dict = None

    def fit(self, X_data, y_data):
        """
        Fit an AdaBoost ensemble model on X_data and y_data.
        Stores the model in the class, in:
            - self.models_dict (dict): includes the models built at each iteration of AdaBoost
            - self.model_weights (dict): includes the alpha weights of each base model

        Args:
            X_data (np.array): 2-d array with the input attributes of training data
            y_data (np.array): 1-d array with target (class) values of training data
        """
        X_data = np.insert(X_data, X_data.shape[1], np.ones(X_data.shape[0]), axis=1)

        self.model_weights = np.zeros(self.n_estimators)
        self.models_dict = {}

        self.domY = len(np.unique(y_data))
        transform = OneHotEncoder(sparse_output=False)
        transform.fit(y_data.reshape(-1, 1))
        self.transform = transform

        for i in range(self.n_estimators):
            # If the model receives a random state, modify it by the corresponding seed (so that it is different at
            # each iteration)
            params = self.params
            if 'random_state' in self.classifier().get_params():
                params['random_state'] = self.random_state + i

            # Create local model with local private data
            model = (self.classifier(**params))

            X_train_no_weight = X_data[:, :-1]
            weights = X_data[:, -1]
            if i != 0:
                d = int(X_data.shape[0])
                prob = weights / weights.sum()
                weighted_choices = np.random.choice(np.arange(d), d, p=prob)
                X_train_no_weight = X_train_no_weight[weighted_choices, :]

            model.fit(X_train_no_weight, y_data)
            self.models_dict[i] = model

            # let's actualize the weights and calculate the model's weight.
            predicted_data = model.predict(X_train_no_weight)
            filter = (y_data == predicted_data)
            err = 1 - accuracy_score(y_data, predicted_data, sample_weight=weights)

            # Check convergence stopping
            if (err < 1 - (1 / self.domY)):
                if (err == 0):
                    alpha = ((math.log(((self.domY - 1) * (1 - (err + 0.005))) / (err + 0.005))) / 2)
                    self.model_weights[i] = alpha
                    break
                else:
                    alpha = ((math.log(((self.domY - 1) * (1 - err)) / err)) / 2)
            else:
                break  #if the error is too high stop iteration
            self.model_weights[i] = alpha

            # When it predicts correctly, weight is decreased
            X_data[filter, -1] = X_data[filter, -1] * math.exp(-alpha)
            # When it predicts wrongly, weight is increased
            X_data[~filter, -1] = X_data[~filter, -1] * math.exp(alpha)

    def predict(self, data, soft_predictions=False):
        """
        Predict class labels for input data, using the previously trained AdaBoost model.
        It uses both the base models and their weights.

        Args:
            data (np.array): Validation data to be predicted

        Return: Array of predicted class labels
        """
        weighted_sum = np.zeros((data.shape[0], self.domY))
        sum_of_weights = 0
        # Get weighted sum of predictions
        for key, model in self.models_dict.items():
            prediction = model.predict(data)
            OneHotprediction = self.transform.transform(prediction.reshape(-1, 1))
            weighted_sum = weighted_sum + OneHotprediction * self.model_weights[key]
            sum_of_weights += self.model_weights[key]

        if soft_predictions:
            return weighted_sum / sum_of_weights
        else:
            predicted_indices = weighted_sum.argmax(axis=1)
            predicted_labels = np.zeros((data.shape[0], self.domY))
            predicted_labels[np.arange(data.shape[0]), predicted_indices] = 1

            return self.transform.inverse_transform(predicted_labels).flatten()
