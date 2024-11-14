# -*- coding: utf-8 -*-
"""
Evaluation utility - AdaBoost-FKD
============

This file contains some utilities for federated model evaluation

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
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from models.AdaBoostFKD import AdaBoostFKD
from models.LocalAdaBoost import LocalAdaBoost


def fl_cross_val_score(data_splits: list, public_data,fl_params={},metric='accuracy',
                        return_overall_scores=False):
    """
    Evaluates the federated model and the local models given by the input parameters 
    across all the data_splits. It also evaluates the centralized models and returns
    a stats report of all splits.

    Args:
        data_splits (list): List of data splits to evaluate the model
        public_data (numpy.ndarray): 2-d array representing the public data used for the knowledge
                distillation process. Such data is not labeled. It must satisfy data.shape[1]==public_data.shape[1]. 
        fl_params (dict): Dictionary containing the parameters used in the FL model.
        metric (str): The metric used to evaluate the performance. It can take the values: 
            -'accuracy': Accuracy metric.
            -'f1_score': f1_score metric.
        return_overall_scores (bool): Wether or not a classification report is returned. Set to False by default.
    Returns:
        FLmodels (list): list of FL models
        centralized_models (list): list of centralized AdaBoost models (with all data)
        accDataFrames (list): list of DataFrame (one for each split) with the scores (according to metric parameter)
          of each client's FL model and local model on a global test and a local test.
        describeDataFrames (list): list of dataframes (one for each split) of score stats (described accDataFrames)
          of FL and local client's models.  
        centralized_model_scores (list): list of the scores of the centralized models (according to metric parameter).
        avg_result (pd.Dataframe): Stats dataframe of the average scores of all client in all splits.

        if return_overall_scores is true, also returns:
            overall_FL_scores (list): list of lists of data_split's clients' FL models classification reports
              (each data_split has a corresponding list of Dataframes with all clients' classification reports
              for that data_split).
            overall_local_scores (list): list of lists of data_split's clients' local models classification reports
              (each data_split has a corresponding list of Dataframes with all clients' classification reports
              for that data_split).
    """

    number_splits = len(data_splits)
    FLmodels = [None] * number_splits
    accDataFrames = [None] * number_splits
    describeDataFrames = [None] * number_splits
    centralized_model_scores = [None] * number_splits
    centralized_models = [None] * number_splits
    overall_FL_scores = [None] * number_splits
    overall_local_scores = [None] * number_splits

    #Take the parameters of the fl_model needed for centralized models:
    try:
        clients_classifier = fl_params['clients_classifier']
    except KeyError:
        clients_classifier = DecisionTreeClassifier
    try:
        clients_classifier_params = fl_params['clients_classifier_params']
    except KeyError:
        clients_classifier_params = {}
    try:
        T = fl_params['T']
    except KeyError:
        T = 10
    try: 
        random_state = fl_params['random_satate']
    except KeyError:
        random_state = 0

    for i, data in enumerate(data_splits):
        # Run all executions with different data but same random seed
        np.random.seed(random_state)
        X_train, X_test, y_train, y_test = data

        # Create and fit the FL model
        FLmodel = AdaBoostFKD(data=X_train, targets=y_train, public_data=public_data,**fl_params)
        FLmodel.fitmodel()

        # Fit the local models (i.e., without using the FL process, but only using local data)
        FLmodel.fit_local_clients_models()

        # Fit an ideal centralized model (i.e., gathering all data together at a single machine)
        centralized_model = LocalAdaBoost(n_estimators=T * 2,classifier=clients_classifier,
                                          params=clients_classifier_params, random_state=random_state)
        centralized_model.fit(X_train, y_train)

        # Compute evaluation metrics for centralized model
        if metric == 'accuracy':
            df = FLmodel.overall_acc_score(X_test, y_test)
            centralized_model_scores[i] = accuracy_score(centralized_model.predict(X_test), y_test)  # * 100
        elif metric == 'f1_score':
            df = FLmodel.overall_F1_score(X_test, y_test)
            centralized_model_scores[i] = f1_score(centralized_model.predict(X_test), y_test,
                                                   labels=np.unique(y_train), average='macro')  # * 100

        # Store models and information of both FL and centralized models
        centralized_models[i] = centralized_model
        FLmodels[i] = FLmodel
        accDataFrames[i] = df
        describeDataFrames[i] = df.describe()

        if return_overall_scores:
            clients_FL_scores = FLmodel.clients_score_dataframes(scheme='FL', global_Xtest=X_test, targets=y_test)
            overall_FL_scores[i] = clients_FL_scores

            clients_local_scores = FLmodel.clients_score_dataframes(scheme='local', global_Xtest=X_test, targets=y_test)
            overall_local_scores[i] = clients_local_scores

    # Compute average results over all data_splits
    avg_result = describeDataFrames[0].loc['mean']
    avg_result.name = 0
    avg_result['centralized_scores'] = centralized_model_scores[0]
    avg_result = pd.Series.to_frame(avg_result)

    for i in range(len(data_splits) - 1):
        a = describeDataFrames[i + 1].loc['mean']
        a['centralized_scores'] = centralized_model_scores[i + 1]
        avg_result[i + 1] = a
    avg_result = avg_result.T.describe()
    if return_overall_scores:
        return (FLmodels, centralized_models, accDataFrames, describeDataFrames, centralized_model_scores, avg_result,
                overall_FL_scores, overall_local_scores)
    else:
        return FLmodels, centralized_models, accDataFrames, describeDataFrames, centralized_model_scores, avg_result


def fl_simple_cross_val_score(data_splits: list, public_data,fl_params={}):
    """
    Evaluates the federated model given by the input parameters across all the data_splits. It
    is a simplified version of fl_cross_val_score. It does not return any result's stats. It returns
    raw scores of each client's model (FL and local) for accuracy and f1-score metric, as well as the
    accuracy and f1-score of the centralized models.
    

    Args:
        data_splits (list): List of data splits to evaluate the model.
        public_data (numpy.ndarray): 2-d array representing the public data used for the knowledge
                distillation process. Such data is not labeled. It must satisfy data.shape[1]==public_data.shape[1]. 
        fl_params (dict): Dictionary containing the parameters used in the FL model.

    Returns:
        FLmodels (list): list of FL models
        centralized_models (list): list of centralized AdaBoost models (with all data)
        accDataFrames (list): list of DataFrames with the scores (accuracy metric)
          of each client's FL and local model on a global test and a local test.
        F1DataFrames (list): list of DataFrames with the scores (f1-score metric)
          of each client's FL and local model on a global test and a local test.
        acc_centralized_model_scores (list): list of the scores of the centralized models
          (accuracy metric).
        f1_centralized_model_scores (list): list of the scores of the centralized models
          (f1-score metric).
    """
    number_splits = len(data_splits)
    FLmodels = [None] * number_splits
    accDataFrames = [None] * number_splits
    F1DataFrames = [None] * number_splits
    acc_centralized_model_scores = [None] * number_splits
    f1_centralized_model_scores = [None] * number_splits
    centralized_models = [None] * number_splits

    #Take the parameters of the fl_model needed for centralized models:
    try:
        clients_classifier = fl_params['clients_classifier']
    except KeyError:
        clients_classifier = DecisionTreeClassifier
    try:
        clients_classifier_params = fl_params['clients_classifier_params']
    except KeyError:
        clients_classifier_params = {}
    try:
        T = fl_params['T']
    except KeyError:
        T = 10
    try: 
        random_state = fl_params['random_satate']
    except KeyError:
        random_state = 0


    for i, data in enumerate(data_splits):
        # Run all executions with different data but same random seed
        np.random.seed(random_state)

        X_train, X_test, y_train, y_test = data

        # Create and fit the FL model
        FLmodel = AdaBoostFKD(data=X_train, targets=y_train, public_data=public_data, **fl_params)
        FLmodel.fitmodel()

        # Fit the local models (i.e., without using the FL process, but only using local data)
        FLmodel.fit_local_clients_models()

        # Fit an ideal centralized model (i.e., gathering all data together at a single machine)
        centralized_model = LocalAdaBoost(n_estimators=T * 2,classifier=clients_classifier,
                                          params=clients_classifier_params, random_state=random_state)
        centralized_model.fit(X_train, y_train)

        df_acc = FLmodel.overall_acc_score(X_test, y_test)
        acc_centralized_model_scores[i] = accuracy_score(centralized_model.predict(X_test), y_test)  #* 100
        DataFramef1 = FLmodel.overall_F1_score(X_test, y_test)
        f1_centralized_model_scores[i] = f1_score(centralized_model.predict(X_test), y_test, labels=np.unique(y_train),
                                                  average='weighted', zero_division=0.0)  #* 100
        centralized_models[i] = centralized_model
        FLmodels[i] = FLmodel
        accDataFrames[i] = df_acc
        F1DataFrames[i] = DataFramef1

    return FLmodels, centralized_models, accDataFrames, F1DataFrames, acc_centralized_model_scores, f1_centralized_model_scores


def get_wilcoxon_ranks(dataset: pd.DataFrame, verbose: bool = False):
    """
    Performs the Wilcoxon signed-rank test. This non-parametric test is used to compare two related samples, matched
    samples, or repeated measurements on a single sample to assess whether their population mean ranks differ. It is
    an alternative to the paired Student's t-test when the data is not normally distributed.

    The code of this method has been partially obtained and modified from the StaTDS library, that is licensed under
    the GNU General Public License v3.0.
        Original Author: Christian Luna, Antonio R. Moya, Jose Maria Luna, Sebastian Ventura
        Original Source: https://github.com/kdis-lab/StaTDS
        Original License: GNU General Public License v3.0

    Args:
        dataset (pandas.DataFrame): A DataFrame with exactly two columns, each representing a different condition or
            time point for the same subjects.
        verbose (bool, optional): If True, prints the detailed results table.

    Returns:
        r_plus (float): sum of positive rankings of the wilcoxon test
        r_minus (float): sum of negative rankings of the wilcoxon test
    """
    if dataset.shape[1] != 2:
        raise "Error: The test only needs two samples"

    results_table = dataset.copy()
    columns = list(dataset.columns)
    differences_results = dataset[columns[0]] - dataset[columns[1]]
    absolute_dif = differences_results.abs()
    absolute_dif = absolute_dif.sort_values()
    results_wilconxon = {"index": [], "dif": [], "rank": [], "R": []}
    rank = 0.0
    tied_ranges = not (len(set(absolute_dif)) == absolute_dif.shape[0])
    for index in absolute_dif.index:
        if math.fabs(0 - absolute_dif[index]) < 1e-10:
            continue
        rank += 1.0
        results_wilconxon["index"] += [index]
        results_wilconxon["dif"] += [differences_results[index]]
        results_wilconxon["rank"] += [rank]
        results_wilconxon["R"] += ["+" if differences_results[index] > 0 else "-"]

    df = pd.DataFrame(results_wilconxon)
    df = df.set_index("index")
    df = df.sort_index()
    results_table = pd.concat([results_table, df], axis=1)

    tie_sum = 0

    if tied_ranges:
        vector = [abs(i) for i in results_table["dif"]]

        counts = {}
        for number in vector:
            try:
                counts[number] = counts[number] + 1
            except KeyError:
                counts[number] = 1

        ranks = results_table["rank"].to_numpy()
        for index, number in enumerate(vector):
            if counts[number] > 1:
                rank_sum = sum(ranks[i] for i, x in enumerate(vector) if x == number)
                average_rank = rank_sum / counts[number]
                for i, x in enumerate(vector):
                    if x == number:
                        ranks[i] = average_rank
        tie_sizes = np.array(list(counts.values()))
        tie_sum = (tie_sizes ** 3 - tie_sizes).sum()

    if verbose:
        print(results_table)

    r_plus = results_table[results_table.R == "+"]["rank"].sum()
    r_minus = results_table[results_table.R == "-"]["rank"].sum()

    return r_plus, r_minus
