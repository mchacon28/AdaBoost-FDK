# -*- coding: utf-8 -*-
"""
Implementation of Federated Random Forest, for comparison purposes.

Original paper:
    Hauschild A.C., Lemanczyk M., Matschinske J., Frisch T., Zolotareva O., Holzinger A., Baumbach J. & Heider, D.
    (2022).
    Federated Random Forests can improve local performance of predictive models for various healthcare applications.
    Bioinformatics, 38(8), 2278-2286.

It is based on the implementation of the FlexTrees library (https://github.com/FLEXible-FL/flex-trees),
published under the AGPLv3 License.
    Herrera F., Jiménez-López D., Argente-Garrido A., Rodríguez-Barroso N., Zuheros C., Aguilera-Martos I., Bello B.,
    García-Márquez M. & Luzón, M. (2024). FLEX: FLEXible Federated Learning Framework. arXiv preprint arXiv:2404.06127.

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

import random

from flex.data import Dataset
from flex.model import FlexModel
from flex.pool import FlexPool
from flextrees.pool import (
    aggregate_trees_from_rf,
    collect_clients_trees_rf,
    deploy_server_config_rf,
    deploy_server_model_rf,
    set_aggregated_trees_rf,
    train_rf,
    init_server_model_rf
)
from flex.pool.decorators import (
    evaluate_server_model,
    init_server_model,
)
from models.flex_GlobalRandomForest import GlobalRandomForest
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

max_depth = 8
n_estimators = 10


@init_server_model
def init_server_model_rf2(config=None, *args, **kwargs):
    """
    Function to initialize the server model

    Args:
        config (dict, optional): Dict that contains the configuration of the server model. Defaults to None.
    """

    server_flex_model = FlexModel()

    if config is None:
        config = {
            'server_params': {
                'max_depth': max_depth,
                'n_estimators': n_estimators,
            },
            'clients_params': {
                'max_depth': max_depth,
                'n_estimators': n_estimators,
            }
        }

    server_flex_model['model'] = GlobalRandomForest(max_depth=config['server_params']['max_depth'],
                                                    n_estimators=config['server_params']['n_estimators'])

    server_flex_model.update(config)

    return server_flex_model


@init_server_model
def init_server_model_rf_theirs(config=None, *args, **kwargs):
    """
    Function to initialize the server model

    Args:
        config (dict, optional): Dict that contains the configuration of the server model. Defaults to None.
    """

    server_flex_model = FlexModel()

    if config is None:
        config = {
            'server_params': {
                'max_depth': 8,
                'n_estimators': 100,
            },
            'clients_params': {
                'max_depth': 8,
                'n_estimators': 100,
            }
        }

    server_flex_model['model'] = GlobalRandomForest(max_depth=config['server_params']['max_depth'],
                                                    n_estimators=config['server_params']['n_estimators'])

    server_flex_model.update(config)

    return server_flex_model


@init_server_model
def init_server_model_rf_no_pruning(config=None, *args, **kwargs):
    """
    Function to initialize the server model
    TO-DO: Completar la descripción de estos 3 métodos, indicando (si sabes, si no nada), la diferencia entre ellos;
    i.e., el rf2, theirs, y no_pruning.

    Args:
        config (dict, optional): Dict that contains the configuration of the
        server model. Defaults to None.
    """

    server_flex_model = FlexModel()

    if config is None:
        config = {
            'server_params': {
                'max_depth': None,
                'n_estimators': 100,
            },
            'clients_params': {
                'max_depth': None,
                'n_estimators': 100,
            }
        }

    server_flex_model['model'] = GlobalRandomForest(max_depth=config['server_params']['max_depth'],
                                                    n_estimators=config['server_params']['n_estimators'])

    server_flex_model.update(config)

    return server_flex_model


@evaluate_server_model
def evaluate_global_rf_model2(server_flex_model, test_data, macro=False, *args, **kwargs):
    """
    Evaluate global model on the server with both locals and a global test set.

    Args:
        server_flex_model (FlexModel): Server Flex Model.
        test_data (ArrayLike): Array with the data to evaluate (both X and y)
        macro (bool): Set it to True if f1-score and AUC macro average wants to be calculated in addition to weighted
            average. Set to False by default.
    Return: tuple with 6 (or 10 if macro is set to True) values:
        - acc_global: Accuracy on the global test set
        - f1w_global: F1-score on the global test set (weighted average)
        - AUCw_global: AUC on the global test set (weighted average)
        - acc_local: Average accuracy on the local tests set of every client
        - f1w_local: Average f1-score (weighted average) on the local tests set of every client
        - AUCw_local: Average AUC (weighted average) on the local tests set of every client
        If macro is set to True, it also returns:
            - f1ma_global (float): F1-score on the global test set (macro average)
            - AUCma_global (float): AUC on the global test set (macro average)
            - f1ma_local (float): Average f1-score (macro average) on the local tests set of every client
            - AUCma_local (float): Average AUC (macro average) on the local tests set of every client
    """
    test_dict, X_global_test, y_global_test = test_data

    preds_rf = server_flex_model['model'].predict(X_global_test)
    preds_rf_proba = server_flex_model['model'].predict(X_global_test, soft=True, n_classes=len(np.unique(y_global_test)))

    acc_global, f1w_global = accuracy_score(y_global_test, preds_rf), \
        f1_score(y_global_test, preds_rf, labels=np.unique(y_global_test), average='weighted', zero_division=0.0)

    if preds_rf_proba.shape[1] == 2:
        preds_rf_proba = preds_rf_proba[:, 1]

    AUCw_global = roc_auc_score(y_global_test, preds_rf_proba, multi_class='ovr', average='weighted', 
                                labels=np.unique(y_global_test))

    if macro:
        f1ma_global = f1_score(y_global_test, preds_rf, labels=np.unique(y_global_test), average='macro',
                               zero_division=0.0)
        AUCma_global = roc_auc_score(y_global_test, preds_rf_proba, multi_class='ovr', average='macro', 
                                labels=np.unique(y_global_test))

    n_clients = len(test_dict)
    f1w_scores = np.zeros(n_clients)
    acc_scores = np.zeros(n_clients)
    AUCw_scores = np.zeros(n_clients)

    if macro:
        f1ma_scores = np.zeros(n_clients)
        AUCma_scores = np.zeros(n_clients)

    for i, (X_test_local, y_test_local) in enumerate(test_dict.values()):
        y_pred = server_flex_model['model'].predict(X_test_local)
        y_pred_proba = server_flex_model['model'].predict(X_test_local, soft=True, n_classes=len(np.unique(y_global_test)))

        if y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]

        acc_score = accuracy_score(y_test_local, y_pred)
        f1w_score = f1_score(y_test_local, y_pred, labels=np.unique(y_global_test), average='weighted',
                             zero_division=0.0)
        AUCw_score = roc_auc_score(y_test_local, y_pred_proba, multi_class='ovr', average='weighted', 
                                labels=np.unique(y_global_test))

        acc_scores[i] = acc_score
        f1w_scores[i] = f1w_score
        AUCw_scores[i] = AUCw_score

        if macro:
            f1ma_scores[i] = f1_score(y_test_local, y_pred, labels=np.unique(y_global_test), average='macro',
                             zero_division=0.0)
            AUCma_scores[i] = roc_auc_score(y_test_local, y_pred_proba, multi_class='ovr', average='macro', 
                                labels=np.unique(y_global_test))

    acc_local = acc_scores.mean()
    f1w_local = f1w_scores.mean()
    AUCw_local = np.nanmean(AUCw_scores) 

    if macro:
        f1ma_local = f1ma_scores.mean()
        AUCma_local = np.nanmean(AUCma_scores)
        return acc_global, f1w_global, f1ma_global, AUCw_global, AUCma_global, acc_local, f1w_local, f1ma_local, AUCw_local, AUCma_local
    else:
        return acc_global, f1w_global, AUCw_global, acc_local, f1w_local, AUCw_local


def FRF_eval(train_dict, test_dict, X_global_test, y_global_test, hyperparameters="ours", config=None,macro=False):
    """
    Trains and evaluates the FRF model
    
    TO_DO: Revisar el parámetro config y ver por qué no funciona lo comentado. 
    Args:
        train_dict (dict): A dictionary with every clients training data. Keys must be numbered
            from 0 to N_clients-1. The data of client i is stored as i:(X_train_i,y_train_i), where 
            X_train_i and y_train_i are 2-d and 1-d numpy arrays respectively.
        test_dict (dict): A dictionary with every clients test data. Keys must be numbered
            from 0 to N_clients-1. The data of client i is stored as i:(X_test_i,y_test_i), where 
            X_test_i and y_test_i are 2-d and 1-d numpy arrays respectively.
        X_global_test (np.array): Global test data to use as evaluation (input attributes)
        y_global_test (np.array): Global test data to use as evaluation (target/classes)
        hyperparameters (str): A string to select server and clients RF parameters used. It can take the values:
            -'ours': uses a max_depth of 8 and 10 estimators for each RF.
            -'theirs': uses a max_depth of 8 and 100 estimators for each RF.
            -'no_pruning': uses a max_depth of None and 100 estimators for each RF. 
            -'other': configurable config. follows the config given in the argument config.
        config (dict): ...
        macro (bool): Set it to True if f1-score and AUC macro average wants to be calculated in addition to weighted
            average. Set to False by default.
         Note: train_dict and test_dict are automatically generated when initializing AdaBoostFDK.

    Return: tuple with 6 (or 10 if macro_f1 is set to True) values:
        - acc_global: Accuracy on the global test set
        - f1w_global: F1-score on the global test set (weighted average)
        - AUCw_global: AUC on the global test set (weighted average)
        - acc_local: Average accuracy on the local tests set of every client
        - f1w_local: Average f1-score (weighted average) on the local tests set of every client
        - AUCw_local: Average AUC (weighted average) on the local tests set of every client
        If macro is set to True, it also returns:
            - f1ma_global: F1-score on the global test set (macro average)
            - AUCma_global: AUC on the global test set (macro average)
            - f1ma_local: Average f1-score (macro average) on the local tests set of every client
            - AUCma_local: Average AUC (macro average) on the local tests set of every client
    """

    n_clients = len(train_dict)
    federated_data = {}

    for key, (data, targets) in train_dict.items():
        flex_data = Dataset.from_array(data, targets)
        federated_data[key] = flex_data
    
    #if hyperparameters == "ours":
    #    configur = {
    #        'server_params': {
    #            'max_depth': 8,
    #            'n_estimators': 10,
    #        },
    #        'clients_params': {
    #            'max_depth': 8,
    #            'n_estimators': 10,
    #        }
    #    }
#
    #    total_estimators = 10
    #elif hyperparameters == "theirs":
    #    configur = {
    #        'server_params': {
    #            'max_depth': 8,
    #            'n_estimators': 100,
    #        },
    #        'clients_params': {
    #            'max_depth': 8,
    #            'n_estimators': 100,
    #        }
    #    }
    #    total_estimators = 100
    #elif hyperparameters == "no_pruning":
    #    configur = {
    #        'server_params': {
    #            'max_depth': None,
    #            'n_estimators': 100,
    #        },
    #        'clients_params': {
    #            'max_depth': None,
    #            'n_estimators': 100,
    #        }
    #    }
    #    total_estimators = 100
    #elif hyperparameters == 'other':
    #    configur = config
    #    try:
    #        total_estimators = configur['server_params']['n_estimators']
    #    except KeyError:
    #        total_estimators=100

    # Set server config
    if hyperparameters == "ours":
        pool = FlexPool.client_server_pool(federated_data, init_server_model_rf2)
        total_estimators = 10
    elif hyperparameters == "theirs":
        pool = FlexPool.client_server_pool(federated_data, init_server_model_rf_theirs)
        total_estimators = 100
    elif hyperparameters == "no_pruning":
        pool = FlexPool.client_server_pool(federated_data, init_server_model_rf_no_pruning)
        total_estimators = 100

    #func = init_server_model_rf(config=configur)
    #pool = FlexPool.client_server_pool(federated_data, func)

    clients = pool.clients
    aggregator = pool.aggregators
    server = pool.servers

    # Total number of estimators

    # Number of estimators per client
    nr_estimators = total_estimators // n_clients

    # Deploy clients config
    server.map(func=deploy_server_config_rf, dst_pool=pool.clients)
    clients.map(func=train_rf)
    #clients.map(func=evaluate_local_rf_model_at_clients)
    aggregator.map(func=collect_clients_trees_rf, dst_pool=pool.clients, nr_estimators=nr_estimators)
    aggregator.map(func=aggregate_trees_from_rf)
    aggregator.map(func=set_aggregated_trees_rf, dst_pool=pool.servers)
    server.map(func=deploy_server_model_rf, dst_pool=pool.clients)
    results = server.map(func=evaluate_global_rf_model2, test_data=(test_dict, X_global_test, y_global_test),
                         macro=macro)

    # acc_global, f1w_global, acc_local, f1w_local = results[0]
    # return acc_global, f1w_global, acc_local, f1w_local

    return results[0]