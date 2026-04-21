# -*- coding: utf-8 -*-
"""
Main AdaBoostFKD algorithm and simulation

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

from flex.data import FedDataDistribution, FedDatasetConfig, Dataset
import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from models.LocalAdaBoost import LocalAdaBoost

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

#For reproducibility, on top of a random_state, fix also a numpy seed (for dirichlet weights)

class AdaBoostFKD:
    """
    Class that implements the Federated adaptation of AdaBoost via ensemble knowledge distillation.
    The operation of the federated scenario is simulated by distribution the data among different clients
        that communicate among them but do not share information.
    """

    def __init__(self, data, targets, public_data,
                 clients_classifier=DecisionTreeClassifier, clients_classifier_params={},
                 server_classifier=DecisionTreeClassifier, server_classifier_params={},
                 n_clients=5, public_data_prediction='majority_voting', server_alpha_weight_adj='common_abs',
                 prediction_weights='only_server', client_weight_adj ='own',soft_predictions=False,temperature=1.0,T=10,
                data_distribution='iid', distribution_param=None,
                 alpha_counter=3, random_state=0, adapt_client_weight=None, balanced_target_client_weight=False,
                 sample_public_data=False, attack_simulation=0.0):
        """
        Args:
            data (numpy.ndarray): A 2-d array representing all the clients' data.
            targets (numpy.ndarray): 1-d array representing true label values associated with data.
            public_data (numpy.ndarray): 2-d array representing the public data used for the knowledge
                distillation process. Such data is not labeled. It must satisfy data.shape[1]==public_data.shape[1]. 
            clients_classifier (class): The learning algorithm (from sklearn) used to train clients' models. At this moment, all
                the clients are supposed to use the same algorithm, but it could be changed in the future.
            clients_classifier_params (dict): Parameters for the clients_classifier class
            server_classifier (class): The learning algorithm (from sklearn) used to train the server's models.
            server_classifier_params (dict): Parameters for the server_classifier
            n_clients (int): Number of clients in the federated scenario
            public_data_prediction (str): It is the way public data is labeled on the server's side, based on each clients'
                model prediction. It can take the values: 'majority_voting', 'weighted_majority_voting'.
            server_alpha_weight_adj (str): It is the way server's models weights are adjusted. It can take the values:
                'common_abs': It uses the mean of all clients' error.
                'common_weighted': It uses the weighted mean (over data instances) of all clients errors.
                'own': Each client has its own weight associated to the server calculated with its own error.
                'avg_abs': mean between 'own' and 'common_abs'
                'avg_weighted': mean between 'own' and 'common_weighted'
            prediction_weights (str): It represents which prediction models are used. It can take the values:
                'only_server': The prediction for each client only uses server models with its weights to predict
                'server_and_clients': The prediction for each client uses the server models and its own local models,
                    each with its corresponding weights.
            client_weight_adj (str): It represents how the clients' weights are adapted during training. It can take the values:
                'common': All clients' weights are adapted the same way, with the server alpha calculated with the server
                error avg (given by server_alpha_weight_adj method).
                'own': Each client's weights are adapted with the server alpha calculated with the error 
                of the server model on its own data.
            adapt_client_weight (numpy.ndarray): It is a 1-d numpy array that takes values in (0,1). It represents
                the scalar that will be multiplied to the clients weights if prediction_weights is 'server_and_clients'.
                Must be of dimension n_clients. If no value is passed, it is set to (d_i/sum(d_i))_n by default,
                where di is the number of data of client i.
            balanced_target_client_weight (bool): If True, it weights the adaptive client weights to unbalanced classes.
                This way, clients with datasets with mostly one class label will not vote much into the FL prediction.
            soft_predictions (bool): If True, it uses soft predictions (i.e., probabilities) instead of hard predictions (i.e., class labels).
            temperature (float): Temperature parameter for softmax function. It is used to control the sharpness of the predictions.
            T (int): Number of communications (i.e., federated rounds) between clients and server.
            data_distribution (str): The way data is distributed between clients. It can take the values:
                'iid': data is distributed i.i.d. between clients.
                'niid_dirichlet_label_skew': Following a Dirichlet distribution per target class. Draws a dirichlet
                    distribution p_n for each clas n so that client j has p_jn proportion of class n.
                'niid_label_quantity_skew': Each client has only data_parameter of the total labels of the target.
                'niid_quantity_skew': Data is partitioned through clients following a Dir_n(data_param) distribution.
            distribution_param (str): The parameter of the distribution followed. It depends on the data_distribution value:
                'iid': no need of parameter
                'niid_dirichlet_label_skew': beta parameter of dirichlet distribution, Dir_n(beta). As beta tends to 0
                    it leads to a more unbalanced partitioning. Must be greater than 0.
                'niid_label_quantity_skew': The amounts of labels per client. Must be greater than 0.
                'niid_quantity_skew': Data distribution according to Dir(beta). Must be greater than 0.
            alpha_counter (int): How many negative server weight in a row stands before the algorithm stops
            random_state (int): Random seed for reproducibility of results
            sample_public_data (bool): Indicates if the public data is sampled each federated round to increase
                diversity.
            attack_simulation (float): Indicates which percentage of the clients is suppossed to attack the model, sending
                random predictions to the server. By default, no client attempts to attack it. The attackers also identify
                their models as well performing, and those of the rest of clients as the worst performance so far.
        """
        self.data = data
        self.targets = targets
        self.n_clients = n_clients
        self.clients_classifier = clients_classifier
        self.clients_classifier_params = clients_classifier_params
        self.server_classifier = server_classifier
        self.server_classifier_params = server_classifier_params
        self.public_data = public_data
        self.public_data_prediction = public_data_prediction
        self.server_alpha_weight_adj = server_alpha_weight_adj
        self.client_weight_adj = client_weight_adj
        self.prediction_weights = prediction_weights
        self.soft_predictions = soft_predictions
        self.temperature = temperature
        self.T = T
        self.data_distribution = data_distribution
        self.distribution_param = distribution_param
        self.alpha_counter = alpha_counter
        self.random_state = random_state
        self.target_values, self.target_count = np.unique(targets, return_counts=True, axis=0)
        self.balanced_target_client_weight = balanced_target_client_weight
        self.domY = len(self.target_values)  #Axis=0 works for OneHot and for 1d array
        self.sample_public_data = sample_public_data
   
        # Choose who are the attackers to the model
        self.attackers = random.sample(list(range(n_clients)), int(attack_simulation*n_clients))

        self.initialize_data()
        if prediction_weights == 'server_and_clients':
            if adapt_client_weight is None:
                self.adapt_client_weight = self.number_data_clients / self.number_total_train_data
            else:
                self.adapt_client_weight = adapt_client_weight
            if self.balanced_target_client_weight:
                self.adapt_client_weight = self.adapt_client_weight * self.balanced_data_constant

    def initialize_data(self):
        """
        Initializes the data for the simulation
        """

        self.data = np.insert(self.data, self.data.shape[1], np.ones(self.data.shape[0]), axis=1)
        clients_names = range(self.n_clients)
        train_clients_data = {}
        test_clients_data = {}
        number_data_clients = np.zeros(self.n_clients)
        balanced_data_constant = np.zeros(self.n_clients)

        # Transforms the numpy to flex dataset to do the partitions
        flex_train_data = Dataset.from_array(self.data, self.targets)

        if self.data_distribution == 'iid':
            federated_data = FedDataDistribution.iid_distribution(centralized_data=flex_train_data,
                                                                  n_nodes=self.n_clients)
        elif self.data_distribution == 'niid_dirichlet_label_skew':
            # Each target class is distributed following a dirichlet distribution
            weights_per_label = (np.random.dirichlet(np.repeat(self.distribution_param, self.n_clients), self.domY).
                                 transpose())

            # Check that each client has at least 10 instances
            #   (It needs the amount of instances of each class and multiply it to the client weights for each class and sum it)
            #   Just this does not work beacuse it sometimes gets divided in just 1 class
            # We impose that at least two classes has 10 or more instances.
            condition = (weights_per_label * self.target_count).sum(axis=1)
            # Try to obtain a proper distribution a maximum of max_cond_iters times; otherwise, raise exception
            max_cond_iters = 100
            iter = 0
            while (condition < 10).any() and iter < max_cond_iters:
                weights_per_label = (np.random.dirichlet(np.repeat(self.distribution_param, self.n_clients), self.domY).
                                     transpose())
                condition = (weights_per_label * self.target_count).sum(axis=1)
                iter += 1
            if (condition < 10).any():
                raise Exception('A proper distribution could not be simulated')

            self.distribution_weights = weights_per_label
            config_niid = FedDatasetConfig(seed=self.random_state, n_nodes=self.n_clients,
                                           replacement=False, weights_per_label=weights_per_label)

            # Get simulated data
            federated_data = FedDataDistribution.from_config(centralized_data=flex_train_data, config=config_niid)
        elif self.data_distribution == 'niid_label_quantity_skew':
            # Each client gets labels_per_node < domY classes of the target values.
            config_niid = FedDatasetConfig(seed=self.random_state, n_nodes=self.n_clients,
                                           replacement=False, labels_per_node=self.distribution_param)

            federated_data = FedDataDistribution.from_config(centralized_data=flex_train_data, config=config_niid)
        elif self.data_distribution == 'niid_quantity_skew':
            # Variable amount of data per client (following a dirichlet distrib)
            weights = np.random.dirichlet(np.repeat(self.distribution_param, self.n_clients))
            # In order to have at least 10 instances per client. The more the instances, the easier to be satisfied
            min_weight_value = 10 / self.data.shape[0]

            # Try to obtain a proper distribution a maximum of max_cond_iters times; otherwise, raise exception
            max_cond_iters = 1000
            iter = 0
            while (weights < min_weight_value).any() and iter < max_cond_iters:
                weights = np.random.dirichlet(np.repeat(self.distribution_param, self.n_clients))
                iter += 1
            if (weights < min_weight_value).any():
                raise Exception('A proper distribution could not be simulated')

            self.distribution_weights = weights
            config_niid = FedDatasetConfig(seed=self.random_state, n_nodes=self.n_clients,
                                           replacement=False, weights=weights)

            federated_data = FedDataDistribution.from_config(centralized_data=flex_train_data, config=config_niid)
        else:
            raise Exception('Data distribution not supported.')

        # Separate data for clients and storing data distribution
        for i in clients_names:
            data, targets = federated_data[i].to_numpy()
            seed = self.random_state + i
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=seed)
            #Change dataset to float type to get more precision on weights' actualization
            X_train = X_train.astype(float)
            X_test = X_test.astype(float)
            train_clients_data[i] = (X_train, y_train)
            test_clients_data[i] = (X_test, y_test)
            number_data_clients[i] = len(y_train)
            _, counts = np.unique(y_train, return_counts=True)
            sorted_count = np.sort(counts)[-2:]
            balanced_data_constant[i] = (sorted_count[-1] - (sorted_count[-1] - sorted_count[0])) / sorted_count[-1]

        self.train_clients_data = train_clients_data
        self.test_clients_data = test_clients_data
        self.number_data_clients = number_data_clients
        self.number_total_train_data = (number_data_clients.sum())
        self.balanced_data_constant = balanced_data_constant

        # OneHotEncoder for weighted predictions
        transform = OneHotEncoder(sparse_output=False)
        transform.fit(self.targets.reshape(-1, 1))
        self.transform = transform

    def public_data_predict(self, arr, use_soft_proba=False):
        """
        Makes the prediction on the public dataset at the server side, via the ensemble knowledge distillation process,
            given the predictions of the clientes on the public data.

        Args:
            arr: Input predictions from clients. Format depends on use_soft_proba:
                - If use_soft_proba=False: 2-d array with shape (n_clients, n_samples). Each row 
                  represents the hard label predictions of a client.
                - If use_soft_proba=True: 3-d array with shape (n_clients, n_samples, n_classes). 
                  Each element [i, :, :] represents the softmax probabilities from client i,
                  already aligned to all classes via complete_log_proba and converted to 
                  probabilities via log_proba_to_softmax.
            use_soft_proba (bool): If False, arr contains hard labels. If True, arr contains 
                softmax probabilities aligned to all classes. Default is False.

        Returns: 1-d array representing the labels assigned to public_data based on the clients' predictions.
        """

        if use_soft_proba:
            # arr has shape (n_clients, n_samples, n_classes)
            n_samples = arr.shape[1]
            average_public_data_predict = np.zeros(n_samples)

            if self.public_data_prediction == 'majority_voting':
                # aggregate probas, then convert soft probas to hard predictions
                agg_probs = np.sum(arr,axis=0) #Shape (n_samples, n_classes)
                average_public_data_predict = np.argmax(agg_probs, axis=1)  # Shape: (n_samples)

            elif self.public_data_prediction == 'weighted_majority_voting':
                # Aggregate softmax probabilities with client weights, then convert to hard labels
                weighted_prob_sum = np.zeros((n_samples, self.domY))
                for i in range(self.n_clients):
                    client_weight = self.number_data_clients[i] / self.number_total_train_data
                    weighted_prob_sum += arr[i] * client_weight

                predicted_indices = weighted_prob_sum.argmax(axis=1)
                predicted_labels = np.zeros((n_samples, self.domY))
                predicted_labels[np.arange(n_samples), predicted_indices] = 1
                average_public_data_predict = self.transform.inverse_transform(predicted_labels).flatten()

        else:
            # Original hard label logic
            #arr has shape (n_clients, n_samples)
            average_public_data_predict = np.zeros(arr.shape[1])

            if self.public_data_prediction == 'majority_voting':
                for i in range(arr.shape[1]):
                    unique, counts = np.unique(arr[:, i], return_counts=True)
                    average_public_data_predict[i] = unique[np.argmax(counts)]
                    # If two predictions were voted the same number of times it takes the one with smaller index

            elif self.public_data_prediction == 'weighted_majority_voting':
                weighted_sum = np.zeros((arr.shape[1], self.domY))
                for i in range(self.n_clients):
                    one_hot_prediction = self.transform.transform(arr[i].reshape(-1, 1))
                    weighted_sum = weighted_sum + (
                            one_hot_prediction * self.number_data_clients[i]) / self.number_total_train_data

                predicted_indices = weighted_sum.argmax(axis=1)
                predicted_labels = np.zeros((arr.shape[1], self.domY))
                predicted_labels[np.arange(arr.shape[1]), predicted_indices] = 1
                average_public_data_predict = self.transform.inverse_transform(predicted_labels).flatten()

        return average_public_data_predict
    
    def log_proba_to_softmax(self, log_proba, temperature=1.0):
        """
        Converts log probabilities to softmax probabilities with temperature scaling.
        
        The temperature parameter controls the "softness" of the probability distribution:
        - temperature < 1.0: sharper distribution (more confident predictions)
        - temperature = 1.0: standard softmax (original probabilities)
        - temperature > 1.0: softer distribution (more uniform, less confident)
        
        This is useful for knowledge distillation where softer targets can provide
        more information about the relationships between classes.
        
        Note: -inf values (unseen classes) will result in probability 0 after softmax,
        as exp(-inf) = 0. The relative magnitudes between finite values are preserved
        since softmax is shift-invariant.
        
        Args:
            log_proba (numpy.ndarray): 2-d array of shape (n_samples, n_classes) containing 
                complete log probabilities (with -inf for unseen classes).
            temperature (float): Temperature parameter for softmax scaling. Default is 1.0.
        
        Returns:
            soft_proba (numpy.ndarray): 2-d array of shape (n_samples, n_classes) with 
                softmax probabilities. Each row sums to 1.
        """
        # Scale log probabilities by temperature
        scaled_log_proba = log_proba / temperature
        
        # For numerical stability, subtract max of FINITE values only
        # This prevents overflow while preserving relative magnitudes
        # (softmax is shift-invariant: softmax(x) = softmax(x - c))
        finite_mask = np.isfinite(scaled_log_proba)
        max_finite = np.where(
            finite_mask, 
            scaled_log_proba, 
            -np.inf
        ).max(axis=1, keepdims=True)
        
        # Handle edge case where all values are -inf (set max to 0)
        max_finite = np.where(np.isinf(max_finite), 0.005, max_finite)
        
        # Compute exp(scaled - max)
        # Note: exp(-inf - anything) = exp(-inf) = 0, so -inf correctly becomes 0
        exp_proba = np.exp(scaled_log_proba - max_finite)
        
        # Normalize to get probabilities
        soft_proba = exp_proba / exp_proba.sum(axis=1, keepdims=True)
        
        return soft_proba

    def complete_log_proba(self, log_proba, model_classes, total_classes=None):
        """
        Completes an incomplete log_proba prediction by adding log(0) = -inf for unseen classes.
        
        This is useful when models are trained on different subsets of classes. The log_proba array
        returned by the model only contains probabilities for classes it has seen. This function
        extends it to include all possible classes with log(0) = -inf for unseen classes.
        
        Args:
            log_proba (numpy.ndarray): 2-d array of shape (n_samples, n_model_classes) containing 
                log probabilities from the model for its trained classes.
            model_classes (numpy.ndarray or list): 1-d array/list of class labels the model was trained on.
                These are typically obtained via model.classes_.
            total_classes (numpy.ndarray or list, optional): 1-d array/list of all possible class labels.
                If None, defaults to self.transform.categories_[0] (all classes seen during fit).
        
        Returns:
            complete_log_proba (numpy.ndarray): 2-d array of shape (n_samples, n_total_classes) with 
                log probabilities for all classes. Unseen classes have log(0) = -inf.
        """
        if total_classes is None:
            total_classes = self.transform.categories_[0]
        
        n_samples = log_proba.shape[0]
        n_total_classes = len(total_classes)
        
        # Initialize with -inf for all classes
        complete_log_proba = np.full((n_samples, n_total_classes), -np.inf)
        
        # Create a mapping from class label to index in total_classes
        # This handles unsorted model_classes correctly
        total_classes_list = list(total_classes)
        for i, model_class in enumerate(model_classes):
            idx = total_classes_list.index(model_class)
            complete_log_proba[:, idx] = log_proba[:, i]
        
        return complete_log_proba

    def client_error_calculation(self, i, j):
        """
        It calculates the error made by the server model and the client's j model at round i on client's j training data. 

        Args:
            i (int): Round
            j (int): Client number or id

        Returns:
            server_err (float): error made by the server model at round i on client's j training data.
            client_err (float): error made by the client's j model at round i on client's j training data.
            alpha (float): float used to update client's j training data weights. Depends on server_err. 
            server_filter (np.array): Boolean array indicating which instances were correctly predicted by the server model
            for client j.
        """
        if False: # j in self.attackers:
            # Says that the server model makes a great error on its data; its own model makes no error on its data
            return 1, 0, self.train_clients_data[j], 1
        else:
            X_train, y_train = self.train_clients_data[j]
            X_train_no_weights = X_train[:, :-1]

            server_model = self.models_dict['server'][i]
            client_model = self.models_dict[j][i]

            server_predicted_data = server_model.predict(X_train_no_weights)
            client_predicted_data = client_model.predict(X_train_no_weights)

            server_filter = (y_train == server_predicted_data)
            sample_weight = X_train[:, -1]

            client_err = 1 - accuracy_score(y_train, client_predicted_data, sample_weight=sample_weight)
            server_err = 1 - accuracy_score(y_train, server_predicted_data, sample_weight=sample_weight)

            if server_err != 1:  # When error is 1 we simply set alpha to 0
                if server_err == 0:
                    # Original AdaBoost stops when the error is 0. Since we have multiple clients we can't do that.
                    # To avoid dividing by 0 we soften the error by adding 0.005; so the weight associated to the model
                    # is high.
                    alpha = ((math.log(((self.domY - 1) * (1 - (server_err + 0.005))) / (server_err + 0.005))) / 2)
                else:
                    alpha = ((math.log(((self.domY - 1) * (1 - server_err)) / server_err)) / 2)
            else:
                alpha = 0

            # If err is greater than threshold, then alpha changes signs and messes the weight actualization
            #if (server_err < 1 - (1 / self.domY)) and (server_err != 0):
            #    # When it predicts correctly, weights are not modified
            #    # When it predicts wrongly, weights are increased
            #    X_train[~server_filter, -1] = X_train[~server_filter, -1] * math.exp(alpha)

            return server_err, client_err, alpha,server_filter
        
    def client_weight_adjustment(self, j,alpha,server_filter,server_err):
        '''
        It reweights client's j weights according to an alpha (calculated in a similar manner to that of AdaBoost algorithm),
        that depends on the server error.
        
        Args:
        param j: Client number or id
        param alpha: float used to update client's j training data weights. Depends on server_err. It can either be the common
        alpha for all clients, or the own alpha of client j.
        param server_filter: np.array: Boolean array indicating which instances were correctly predicted by the server model
        for client j.
        param server_err: error made by the server model used to calculate alpha.

        returns:
        (X_train, y_train) (tuple of np.arrays): Tuple of 2d and 1d arrays representing client's j
            training data with its weights actualized.
        '''
        X_train, y_train = self.train_clients_data[j]

        if (server_err < 1 - (1 / self.domY)) and (server_err != 0):
                # When it predicts correctly, weights are not modified
                # When it predicts wrongly, weights are increased
                X_train[~server_filter, -1] = X_train[~server_filter, -1] * math.exp(alpha)
        return (X_train,y_train)

    def server_alpha_weight_adjustment(self, err_arr):
        """
        Calculate the alpha weight of the server model based on the client's errors.

        Args:
            err_arr (np.array): Error of the server model on each client's data (so the length of err_arr
            is the same as the number of clients).

        Returns: Weight alpha of the server model, corresponding to such errors.
        """

        if (self.server_alpha_weight_adj == 'common_abs') or (self.server_alpha_weight_adj == 'avg_abs'):
            server_err = (err_arr.sum()) / self.n_clients
        elif ((self.server_alpha_weight_adj == 'common_weighted') or
                (self.server_alpha_weight_adj == 'avg_weighted') or
                (self.server_alpha_weight_adj == 'own')):
            # Even though own doesn't need these weights they are calculated to have a stopping criteria.
            server_err = ((err_arr * self.number_data_clients).sum()) / self.number_total_train_data
        else:
            raise Exception('Aggregation of client and server model weights is not supported.')

        if server_err != 1:  # When error is 1 we simply set alpha to 0
            if server_err == 0:
                # Original AdaBoost stops when the error is 0. Since we have multiple clients we can't do that.
                # To avoid dividing by 0 we soften the error by adding 0.005; so the weight associated to the model
                # is high.
                alpha = ((math.log(((self.domY - 1) * (1 - (server_err + 0.005))) / (server_err + 0.005))) / 2)
            else:
                alpha = ((math.log(((self.domY - 1) * (1 - server_err)) / server_err)) / 2)
        else:
            alpha = 0

        return alpha,server_err

    def fitmodel(self):
        """
        Simulate the federated model training
        """
        # public_data = self.public_data
        # n_clients = self.n_clients

        # Stores public data predictions of each client (each row is a client)
        predicted_public_data = np.zeros((self.n_clients, self.public_data.shape[0]))

        self.models_dict = {}

        # A counter to check how many times in a row server's alpha weight is zero.
        # When it gets to 3 iterations, the learning stops.
        counter = 0

        for i in range(self.n_clients):
            self.models_dict[i] = {}
        self.models_dict['server'] = {}

        self.server_models_weights = np.zeros(self.T)
        # stores in (i,j) each alpha of j-th client, calculated with the error of the server model at i-th round
        self.own_server_model_weights = np.zeros((self.T, self.n_clients))
        self.clients_model_weights = np.zeros((self.T, self.n_clients))

        # T federated rounds (i.e. communications)
        for i in range(self.T):
            # Indicate which public data to use
            if self.sample_public_data:
                # Sample a different subset (e.g. 80%) of public data each iteration, to increase diversity
                curr_public_data, _ = train_test_split(self.public_data, train_size=0.8,
                                                       random_state=self.random_state+i)

                # Sample with replacement of the original public data each iteration, to increase diversity
                # sample_indexes = np.random.choice(self.public_data.shape[0], self.public_data.shape[0], replace=True)
                # curr_public_data = self.public_data[sample_indexes, :]
            else:
                # Use the whole public data each iteration
                curr_public_data = self.public_data

            if self.soft_predictions:
                predicted_public_data = np.zeros((self.n_clients, curr_public_data.shape[0], self.domY))
            else:
                predicted_public_data = np.zeros((self.n_clients, curr_public_data.shape[0]))

            # CLIENTS SIDE
            for j in range(self.n_clients):
                seed = self.random_state + (j+1)*(i+1)

                # If the model has a random state, modify it by the corresponding seed (so that it is different in
                # each client, and at each round)
                params = self.clients_classifier_params
                if 'random_state' in self.clients_classifier().get_params():
                    params['random_state'] = seed

                # Create local model with local private data
                model = (self.clients_classifier(**params))
                X_train, y_train = self.train_clients_data[j]
                X_train_no_weight = X_train[:, :-1]
                y_train_no_weight = y_train[:]
                weights = X_train[:, -1]
                if i != 0:
                    d = int(self.number_data_clients[j])
                    prob = weights / weights.sum()
                    weighted_choices = np.random.choice(np.arange(d), d, p=prob)
                    X_train_no_weight = X_train_no_weight[weighted_choices, :]
                    y_train_no_weight = y_train[weighted_choices]

                if j in self.attackers:
                    # predicted_public_data[j, :] = [random.choice(np.unique(self.train_clients_data[j][1])) for _ in range(len(curr_public_data))]
                    # Fit local model with random labels from y
                    model.fit(X_train_no_weight, [random.choice(np.unique(y_train_no_weight)) for _ in range(len(y_train_no_weight))])
                else:
                    # Fit local model
                    model.fit(X_train_no_weight, y_train_no_weight)

                # Get predictions over public unlabeled data
                if self.soft_predictions:
                    log_proba= model.predict_log_proba(curr_public_data)
                    #print(log_proba)
                    complete_log_proba = self.complete_log_proba(log_proba, model.classes_)
                    #print(complete_log_proba)
                    predicted_public_data[j, :, :] = self.log_proba_to_softmax(complete_log_proba,temperature=self.temperature)
                    #print(predicted_public_data[j, :, :])
                else:
                    predicted_public_data[j, :] = model.predict(curr_public_data)

                self.models_dict[j][i] = model

            # SERVER SIDE
            # Get aggregated predictions over public dataset
            voted_public_data_labels = self.public_data_predict(predicted_public_data,use_soft_proba=self.soft_predictions)

            # Create and build model at the server side over the public dataset
            params = self.server_classifier_params
            if 'random_state' in self.server_classifier().get_params():
                params['random_state'] = self.random_state + self.n_clients*(i+1)
            server_model = self.server_classifier(**params)
            server_model.fit(curr_public_data, voted_public_data_labels)
            self.models_dict['server'][i] = server_model

            # Compute error of the server model at both server's and clients' side
            server_err_arr = np.zeros(self.n_clients)
            clients_err_arr = np.zeros(self.n_clients)
            server_filter = {j:None for j in range(self.n_clients)}
            for j in range(self.n_clients):
                server_err, client_err, own_client_alpha,server_filter[j] = \
                    self.client_error_calculation(i, j)
                server_err_arr[j] = server_err
                clients_err_arr[j] = client_err
                # Clients own alpha with its data and client model
                self.own_server_model_weights[i, j] = own_client_alpha

            # shared alpha weights for all clients (mean of errors of server model predicting client data)
            alpha,aggregated_server_error = self.server_alpha_weight_adjustment(server_err_arr)
            if self.client_weight_adj == 'common':
                for j in range(self.n_clients):    
                    self.train_clients_data[j] = self.client_weight_adjustment(j,alpha,server_filter[j],aggregated_server_error)
            elif self.client_weight_adj == 'own':
                for j in range(self.n_clients):    
                    client_alpha = self.own_server_model_weights[i, j]
                    self.train_clients_data[j] = self.client_weight_adjustment(j,client_alpha,server_filter[j],server_err_arr[j])

            # To avoid division by 0 and get a high value of alpha.
            clients_err_arr[clients_err_arr == 0] = 0.005
            # To avoid log(0) and to not count that model. Setting it to this makes the alpha <=0 hence, 0.
            clients_err_arr[clients_err_arr == 1] = 1 - (1 / self.domY)
            # abs alphas associated with each client
            self.clients_model_weights[i, :] = ((np.log(((self.domY - 1)*(1 - clients_err_arr)) / clients_err_arr)) / 2)

            # Checks if alphas are negative and sets them to zero.
            # If more than self.alpha_counter alphas are negative in a row, the iteration stops.
            if (alpha <= 0):
                if counter < (self.alpha_counter):
                    counter += 1
                else:
                    break
            else:
                counter = 0

            # Set alpha weight for the current server model
            self.server_models_weights[i] = max(alpha, 0)

        # All the clients server weights that were negative are set to zero.
        self.own_server_model_weights[self.own_server_model_weights < 0] = 0
        # All the clients models weights that were negative are set to zero.
        self.clients_model_weights[self.clients_model_weights < 0] = 0

    def fit_local_clients_models(self):
        """
        Fit a local model at the client's side, without considering the federated process

        """
        self.local_clients_models_dict = {}
        for i in range(self.n_clients):
            if False: # i in self.attackers:
                self.local_clients_models_dict[i] = None
            else:
                X_train, y_train = self.train_clients_data[i]
                seed = self.random_state + i
                #The number of estimators is 2T so that the clients on local and with server have the same amount of models.
                model = LocalAdaBoost(n_estimators=self.T * 2,classifier=self.clients_classifier,
                                    params=self.clients_classifier_params, random_state=seed)
                model.fit(X_train[:, :-1], y_train)
                self.local_clients_models_dict[i] = model

    def predict_data(self, data, server_weights, i=None, soft_predictions=False):
        """
        A method that makes the prediction associated with client i of the data given using server_weights array as the
        server weights for the prediction.
        
        Args:
            data (np.array): 2-d array to be labeled.
            server_weights (np.array): 1-d array corresponding to the weights used by the server models. It must be of
            length self.T (number of rounds).
            i (int): Identifier of the client.

        Returns: Prediction of the given client over data.
        """
        # array with as many rows as the number of data, and as many columns as targets array
        weighted_sum = np.zeros((data.shape[0], self.domY))
        sum_of_weights = 0
        if self.prediction_weights == 'only_server':
            for key, model in self.models_dict['server'].items():
                prediction = model.predict(data)
                one_hot_prediction = self.transform.transform(prediction.reshape(-1, 1))
                weighted_sum = weighted_sum + one_hot_prediction * server_weights[key]
                sum_of_weights += server_weights[key]
        elif self.prediction_weights == 'server_and_clients':
            for key, model in self.models_dict['server'].items():
                server_prediction = model.predict(data)
                client_prediction = self.models_dict[i][key].predict(data)
                server_one_hot_prediction = self.transform.transform(server_prediction.reshape(-1, 1))
                client_one_hot_prediction = self.transform.transform(client_prediction.reshape(-1, 1))
                weighted_sum = weighted_sum + server_one_hot_prediction * server_weights[key]
                weighted_sum = weighted_sum + client_one_hot_prediction * (
                        self.clients_model_weights[key, i] * self.adapt_client_weight[i])
                sum_of_weights += server_weights[key] + self.clients_model_weights[key, i] * self.adapt_client_weight[i]
        if soft_predictions:
            return weighted_sum / sum_of_weights
        else:
            predicted_indices = weighted_sum.argmax(axis=1)
            predicted_labels = np.zeros((data.shape[0], self.domY))
            predicted_labels[np.arange(data.shape[0]), predicted_indices] = 1

            return self.transform.inverse_transform(predicted_labels).flatten()

    def client_predict_data(self, data, i, soft_predictions=False):
        """
        Predicts the labels of data associated with client i. Unlike self.predict_data, this method internally
        handdles which server_weights to choose.

        Args:
            data (np.array): 2-d array to be labeled.
            i (int): Identifier of the client.

        Returns: Prediction of the given client over data.
        """
        if (self.server_alpha_weight_adj == 'common_abs') or (self.server_alpha_weight_adj == 'common_weighted'):
            if self.prediction_weights == 'only_server':
                predicted_data = self.predict_data(data, self.server_models_weights, soft_predictions=soft_predictions)
            elif self.prediction_weights == 'server_and_clients':
                predicted_data = self.predict_data(data, self.server_models_weights, i, soft_predictions=soft_predictions)
        elif self.server_alpha_weight_adj == 'own':
            weights = self.own_server_model_weights[:, i]
            predicted_data = self.predict_data(data, weights, i, soft_predictions=soft_predictions)
        elif (self.server_alpha_weight_adj == 'avg_abs') or (self.server_alpha_weight_adj == 'avg_weighted'):
            weights = (self.own_server_model_weights[:, i] + self.server_models_weights) / 2
            predicted_data = self.predict_data(data, weights, i, soft_predictions=soft_predictions)

        return predicted_data

    def global_predict_data(self, data):
        """
        Obtain the prediction of the Ensemble FL model

        Args:
            data (np.array): 2-d array to be labeled.

        Returns: 2-d array with the prediction of each client (at each row) over data
        """
        all_predicted_labels = np.zeros((self.n_clients, data.shape[0]))

        if (self.server_alpha_weight_adj == 'common_abs') or (self.server_alpha_weight_adj == 'common_weighted'):
            if self.prediction_weights == 'only_server':
                predicted_data = self.predict_data(data, self.server_models_weights)
                all_predicted_labels[:, :] = predicted_data
            elif self.prediction_weights == 'server_and_clients':
                for i in range(self.n_clients):
                    predicted_data = self.predict_data(data, self.server_models_weights, i)
                    all_predicted_labels[i, :] = predicted_data
        elif self.server_alpha_weight_adj == 'own':
            for i in range(self.n_clients):
                weights = self.own_server_model_weights[:, i]
                predicted_data = self.predict_data(data, weights, i)
                all_predicted_labels[i, :] = predicted_data
        elif (self.server_alpha_weight_adj == 'avg_abs') or (self.server_alpha_weight_adj == 'avg_weighted'):
            for i in range(self.n_clients):
                weights = (self.own_server_model_weights[:, i] + self.server_models_weights) / 2
                predicted_data = self.predict_data(data, weights, i)
                all_predicted_labels[i, :] = predicted_data

        return all_predicted_labels

    def overall_acc_score(self, X_global, y_global):
        """
        Obtain the overall accuracy score of the federated model and the only local ones, using either the client's
            local data and the global data received as parameter.

        Args:
            X_global (np.array): 2-d array to be predicted.
            y_global (np.array): 1-d array representing X_global true labels.

        Returns: Dataframe with a row for each client and columns:
            - data_distrib: Data of each client
            - FL_acc_own_data: Accuracy of the federated model with each client's data
            - FL_acc_global_data: Accuracy of the federated model with the data given as input
            - local_acc_own_data: Accuracy of the local Adaboost model with each client's data
            - local_acc_global_data: Accuracy of the local Adaboost model with data given as input
            - local_difference: Difference between the federated and local models on local data
            - global_difference: Difference between the federated and local models on global data
        """

        FL_acc_own_data = np.zeros(self.n_clients)
        FL_acc_global_data = np.zeros(self.n_clients)
        local_acc_own_data = np.zeros(self.n_clients)
        local_acc_global_data = np.zeros(self.n_clients)
        global_difference = np.zeros(self.n_clients)
        local_difference = np.zeros(self.n_clients)

        for i in range(self.n_clients):
            if i not in self.attackers:
                own_data_Xtest, own_data_ytest = self.test_clients_data[i]
                local_model = self.local_clients_models_dict[i]

                FL_acc_own_data[i] = accuracy_score(self.client_predict_data(own_data_Xtest[:, :-1], i), own_data_ytest)
                FL_acc_global_data[i] = accuracy_score(self.client_predict_data(X_global, i), y_global)
                local_acc_own_data[i] = accuracy_score(local_model.predict(own_data_Xtest[:, :-1]), own_data_ytest)
                local_acc_global_data[i] = accuracy_score(local_model.predict(X_global), y_global)
                global_difference[i] = FL_acc_global_data[i] - local_acc_global_data[i]
                local_difference[i] = FL_acc_own_data[i] - local_acc_own_data[i]
            else:
                # The score of the attacker clients is not considered to assess the performance of the clients' models
                FL_acc_own_data[i] = np.nan
                FL_acc_global_data[i] = np.nan
                local_acc_own_data[i] = np.nan
                local_acc_global_data[i] = np.nan
                global_difference[i] = np.nan
                local_difference[i] = np.nan

        return pd.DataFrame({'data_distrib': self.number_data_clients,
                             'FL_acc_own_data': FL_acc_own_data, 'FL_acc_global_data': FL_acc_global_data,
                             'local_acc_own_data': local_acc_own_data, 'local_acc_global_data': local_acc_global_data,
                             'local_difference': local_difference, 'global_difference': global_difference})

    def overall_F1_score(self, X_global, y_global, macro_f1=False):
        """
        Obtain the overall f1-score score of the federated model and the only local ones, using either the client's
            local data and the global data received as parameter.

        Args:
            X_global (np.array): 2-d array to be predicted.
            y_global (np.array): 1-d array representing X_global true labels.
            macro_f1 (bool): Set it to True if f1-score macro average wants to be calculated in addition to weighted
            average. Set to False by default.  

        Returns: Dataframe with a row for each client and columns:
            - data_distrib: Data of each client
            - FL_f1_own_data: F1-score of the federated model with each client's data
            - FL_f1_global_data: F1-score of the federated model with the data given as input
            - local_wf1_own_data: F1-score of the local Adaboost model with each client's data
            - local_wf1_global_data: F1-score of the local Adaboost model with data given as input
            - local_difference_w: Difference between the federated and local models on local data
            - global_difference: Difference between the federated and global models on global data
        """
        FL_wf1_own_data = np.zeros(self.n_clients)
        FL_wf1_global_data = np.zeros(self.n_clients)
        local_wf1_own_data = np.zeros(self.n_clients)
        local_wf1_global_data = np.zeros(self.n_clients)
        global_difference_w = np.zeros(self.n_clients)
        local_difference_w = np.zeros(self.n_clients)

        if macro_f1:
            FL_maf1_own_data = np.zeros(self.n_clients)
            FL_maf1_global_data = np.zeros(self.n_clients)
            local_maf1_own_data = np.zeros(self.n_clients)
            local_maf1_global_data = np.zeros(self.n_clients)
            global_difference_ma = np.zeros(self.n_clients)
            local_difference_ma = np.zeros(self.n_clients)

        for i in range(self.n_clients):
            if i not in self.attackers:
                own_data_Xtest, own_data_ytest = self.test_clients_data[i]
                local_model = self.local_clients_models_dict[i]

                FL_wf1_own_data[i] = f1_score(self.client_predict_data(own_data_Xtest[:, :-1], i), own_data_ytest,
                                            labels=np.unique(own_data_ytest), average='weighted', zero_division=0.0)
                FL_wf1_global_data[i] = f1_score(self.client_predict_data(X_global, i), y_global,
                                                labels=np.unique(y_global), average='weighted', zero_division=0.0)

                local_wf1_own_data[i] = f1_score(local_model.predict(own_data_Xtest[:, :-1]), own_data_ytest,
                                                labels=np.unique(own_data_ytest), average='weighted', zero_division=0.0)
                local_wf1_global_data[i] = f1_score(local_model.predict(X_global), y_global,
                                                labels=np.unique(y_global), average='weighted', zero_division=0.0)

                global_difference_w[i] = FL_wf1_global_data[i] - local_wf1_global_data[i]
                local_difference_w[i] = FL_wf1_own_data[i] - local_wf1_own_data[i]

                if macro_f1:
                    FL_maf1_own_data[i] = f1_score(self.client_predict_data(own_data_Xtest[:, :-1], i), own_data_ytest,
                                                labels=np.unique(own_data_ytest), average='macro', zero_division=0.0)
                    FL_maf1_global_data[i] = f1_score(self.client_predict_data(X_global, i), y_global,
                                                    labels=np.unique(y_global), average='macro', zero_division=0.0)
                    local_maf1_own_data[i] = f1_score(local_model.predict(own_data_Xtest[:, :-1]), own_data_ytest,
                                                    labels=np.unique(own_data_ytest), average='macro', zero_division=0.0)
                    local_maf1_global_data[i] = f1_score(local_model.predict(X_global), y_global,
                                                        labels=np.unique(y_global), average='macro', zero_division=0.0)
                    global_difference_ma[i] = FL_maf1_global_data[i] - local_maf1_global_data[i]
                    local_difference_ma[i] = FL_maf1_own_data[i] - local_maf1_own_data[i]
            else:
                # The score of the attacker clients is not considered to assess the performance of the clients' models
                FL_wf1_own_data[i] = np.nan
                FL_wf1_global_data[i] = np.nan
                local_wf1_own_data[i] = np.nan
                local_wf1_global_data[i] = np.nan
                global_difference_w[i] = np.nan
                local_difference_w[i] = np.nan

                if macro_f1:
                    FL_maf1_own_data[i] = np.nan
                    FL_maf1_global_data[i] = np.nan
                    local_maf1_own_data[i] = np.nan
                    local_maf1_global_data[i] = np.nan
                    global_difference_ma[i] = np.nan
                    local_difference_ma[i] = np.nan

        if macro_f1:
            return pd.DataFrame(
                {'data_distrib': self.number_data_clients,
                 'FL_wf1_own_data': FL_wf1_own_data, 'FL_wf1_global_data': FL_wf1_global_data,
                 'local_wf1_own_data': local_wf1_own_data, 'local_wf1_global_data': local_wf1_global_data,
                 'local_difference_w': local_difference_w, 'global_difference_w': global_difference_w,
                 'FL_maf1_own_data': FL_maf1_own_data, 'FL_maf1_global_data': FL_maf1_global_data,
                 'local_maf1_own_data': local_maf1_own_data, 'local_maf1_global_data': local_maf1_global_data,
                 'local_difference_ma': local_difference_ma, 'global_difference_ma': global_difference_ma,
                 }
            )
        else:
            return pd.DataFrame(
                {'data_distrib': self.number_data_clients,
                 'FL_wf1_own_data': FL_wf1_own_data, 'FL_wf1_global_data': FL_wf1_global_data,
                 'local_wf1_own_data': local_wf1_own_data, 'local_wf1_global_data': local_wf1_global_data,
                 'local_difference_w': local_difference_w, 'global_difference_w': global_difference_w}
            )

    def clients_score_dataframes(self, scheme, global_Xtest, targets):
        """
        Calculates the scores of each client, depending on the scheme. For each client
        a classification_report like the one in sklearn will be returned.

        Args:
            scheme (str): Models to which perform the classification report. It can 
            take the values:
            'FL': Classification report is made with the FL prediction associated to each client.
            'local': Classification report is made with the prediction of each client's local model.
            
            global_Xtest (np.array): 2-d array representing the data to be predicted and used in
            the classification reports.
            targets (np.array): 1-d array representing the true labels of global_Xtest. 

        Returns: List of dataframes consisting of each of the clients classification report (according 
        to scheme).
        """

        score_models = [None] * self.n_clients
        labels, counts = np.unique(targets, return_counts=True)
        counts = counts.reshape(-1, 1)
        total_counts = np.ones((2, 1)) * len(targets)

        for i in range(self.n_clients):
            if scheme == 'FL':
                predictions = self.client_predict_data(global_Xtest, i)
            elif scheme == 'local':
                local_model = self.local_clients_models_dict[i]
                predictions = local_model.predict(global_Xtest)

            dataframe = pd.DataFrame(np.nan,
                                     index=(labels.tolist() + ['Accuracy', 'macro_avg', 'weighted_avg']),
                                     columns=['precision', 'recall', 'f1_score', 'count'])

            #Scores for each label in test set
            f1score = f1_score(predictions, targets, labels=labels, average=None).reshape(-1, 1)
            recall = recall_score(predictions, targets, labels=labels, average=None).reshape(-1, 1)
            precision = precision_score(predictions, targets, labels=labels, average=None).reshape(-1, 1)
            labels_scores = np.concatenate([precision, recall, f1score, counts], axis=1)

            # Storing the scores of FL
            dataframe.loc[labels] = labels_scores

            #Avg scores for all labels 
            macrof1score = f1_score(predictions, targets, labels=labels, average='macro')
            weightedf1score = f1_score(predictions, targets, labels=labels, average='weighted')
            macrorecall = recall_score(predictions, targets, labels=labels, average='macro')
            weightedrecall = recall_score(predictions, targets, labels=labels, average='weighted')
            macroprecision = precision_score(predictions, targets, labels=labels, average='macro')
            weightedprecision = precision_score(predictions, targets, labels=labels, average='weighted')

            avg_FL_f1score = np.array([macrof1score, weightedf1score]).reshape(-1, 1)
            avg_FL_recall = np.array((macrorecall, weightedrecall)).reshape(-1, 1)
            avg_FL_precision = np.array((macroprecision, weightedprecision)).reshape(-1, 1)
            overall_scores = np.concatenate([avg_FL_precision, avg_FL_recall, avg_FL_f1score, total_counts], axis=1)

            dataframe.iloc[-2:] = overall_scores

            #Accuracy score
            dataframe.loc['Accuracy', ['f1_score', 'count']] = [accuracy_score(predictions, targets), len(targets)]
            dataframe = dataframe.rename_axis(f'Client {i}', axis=1)
            score_models[i] = dataframe

        return score_models

    def score_model(self):
        """
        Obtain a dictionary with keys the number of the clients and values a tuple (clients accuracy,server accuracy)
        on each clients' test

        Returns: dict with the scores
        """
        scores_dict = {}
        for i in range(self.n_clients):
            X_test, y_test = self.test_clients_data[i]
            X_test = X_test[:, :-1]

            # Prediction with the first decision Tree
            client_prediction = self.local_clients_models_dict[i].predict(X_test)
            server_prediction = self.client_predict_data(X_test, i)
            client_acc = accuracy_score(y_test, client_prediction)
            server_acc = accuracy_score(y_test, server_prediction)
            scores_dict[i] = (client_acc, server_acc)

        return scores_dict

    def overall_score(self, X_global, y_global, macro_f1=False):
        """
        A method that calculates the accuracy score and f1 score of all clients FL model, both on a global test (input)
        and on each clients' local tests and returns the average score of each. 
        Args:
            X_global (np.array): 2-d array representing a global test to check the average FL model performance. 
            y_global (np.array): 1-d array representing true X_global labels.
            macro_f1 (bool): If set to True, f1 score is also calculated with a macro average. Set to False by default.

        Returns: Dataframe with a row for each client and columns:
            - acc_global (float): average of every clients' accuracy score result on X_global data. 
            - f1w_global (float): average of every clients' f1 (weighted average) score result on X_global data.
            - rocw_global (float): average of every clients' roc_auc (weighted average) score result on X_global data.
            - acc_local (float): average of every clients' accuracy score result on its own test data.
            - f1w_local (float): average of every clients' f1 (weighted average) score result on its own test data.
            - rocw_local (float): average of every clients' roc_auc (weighted average) score result on its own test data.
            If macro_f1 is set to True, it also returns:
            - f1ma_global (float): average of every clients' f1 (macro average) score result on X_global data.
            - rocma_global (float): average of every clients' roc_auc (macro average) score result on X_global data.
            - f1ma_local (float): average of every clients' f1 (macro average) score result on its own test data.
            - rocma_local (float): average of every clients' roc_auc (macro average) score result on its own test data.
        """
        FL_acc_own_data_wf1 = np.zeros(self.n_clients)
        FL_acc_global_data_wf1 = np.zeros(self.n_clients)
        FL_acc_own_data_acc = np.zeros(self.n_clients)
        FL_acc_global_data_acc = np.zeros(self.n_clients)
        FL_acc_own_data_wroc = np.zeros(self.n_clients)
        FL_acc_global_data_wroc = np.zeros(self.n_clients)

        if macro_f1:
            FL_acc_own_data_maf1 = np.zeros(self.n_clients)
            FL_acc_global_data_maf1 = np.zeros(self.n_clients)
            FL_acc_own_data_maroc = np.zeros(self.n_clients)
            FL_acc_global_data_maroc = np.zeros(self.n_clients)

        for i in range(self.n_clients):
            if i not in self.attackers:
                own_data_Xtest, own_data_ytest = self.test_clients_data[i]
                
                
                # Predictions for Accuracy and F1 (Hard labels)
                y_pred_own = self.client_predict_data(own_data_Xtest[:, :-1], i)
                y_pred_global = self.client_predict_data(X_global, i)

                
                # Predictions for ROC AUC (Soft probabilities)
                y_prob_own = self.client_predict_data(own_data_Xtest[:, :-1], i, soft_predictions=True)
                y_prob_global = self.client_predict_data(X_global, i, soft_predictions=True)

                if y_prob_own.shape[1] == 2:
                    y_prob_own = y_prob_own[:, 1]
                    y_prob_global = y_prob_global[:, 1]
                
                FL_acc_own_data_wf1[i] = f1_score(y_pred_own, own_data_ytest,
                                                labels=np.unique(own_data_ytest), average='weighted', zero_division=0.0)
                FL_acc_global_data_wf1[i] = f1_score(y_pred_global, y_global,
                                                    labels=np.unique(y_global), average='weighted', zero_division=0.0)
                FL_acc_global_data_acc[i] = accuracy_score(y_pred_global, y_global)
                FL_acc_own_data_acc[i] = accuracy_score(y_pred_own, own_data_ytest)

                
                FL_acc_own_data_wroc[i] = roc_auc_score(own_data_ytest, y_prob_own, multi_class='ovr', average='weighted', labels=np.unique(y_global))
                
                FL_acc_global_data_wroc[i] = roc_auc_score(y_global, y_prob_global, multi_class='ovr', average='weighted', labels=np.unique(y_global))

                if macro_f1:
                    FL_acc_own_data_maf1[i] = f1_score(y_pred_own, own_data_ytest,
                                                    labels=np.unique(own_data_ytest), average='macro', zero_division=0.0)
                    FL_acc_global_data_maf1[i] = f1_score(y_pred_global, y_global,
                                                        labels=np.unique(y_global), average='macro', zero_division=0.0)
                    
                    FL_acc_own_data_maroc[i] = roc_auc_score(own_data_ytest, y_prob_own, multi_class='ovr', average='macro', labels=np.unique(y_global))
                    FL_acc_global_data_maroc[i] = roc_auc_score(y_global, y_prob_global, multi_class='ovr', average='macro', labels=np.unique(y_global))

            else:
                FL_acc_own_data_wf1[i] = np.nan
                FL_acc_global_data_wf1[i] = np.nan
                FL_acc_global_data_acc[i] = np.nan
                FL_acc_own_data_acc[i] = np.nan
                FL_acc_own_data_wroc[i] = np.nan
                FL_acc_global_data_wroc[i] = np.nan

                if macro_f1:
                    FL_acc_own_data_maf1[i] = np.nan
                    FL_acc_global_data_maf1[i] = np.nan
                    FL_acc_own_data_maroc[i] = np.nan
                    FL_acc_global_data_maroc[i] = np.nan

        acc_global = np.nanmean(FL_acc_global_data_acc)
        acc_local = np.nanmean(FL_acc_own_data_acc)
        f1w_global = np.nanmean(FL_acc_global_data_wf1)
        f1w_local = np.nanmean(FL_acc_own_data_wf1)
        rocw_global = np.nanmean(FL_acc_global_data_wroc)
        rocw_local = np.nanmean(FL_acc_own_data_wroc)

        if macro_f1:
            f1ma_global = np.nanmean(FL_acc_global_data_maf1)
            f1ma_local = np.nanmean(FL_acc_own_data_maf1)
            rocma_global = np.nanmean(FL_acc_global_data_maroc)
            rocma_local = np.nanmean(FL_acc_own_data_maroc)
            return acc_global, f1w_global, f1ma_global, rocw_global, rocma_global, acc_local, f1w_local, f1ma_local, rocw_local, rocma_local
        else:
            return acc_global, f1w_global, rocw_global, acc_local, f1w_local, rocw_local
