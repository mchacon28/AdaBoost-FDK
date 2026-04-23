# -*- coding: utf-8 -*-
"""
Implementation of GlobalRandomForest, for comparison purposes.

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
    2026-04-23

Version:
    1.0.1

License:
    GNU General Public License v3
"""
from copy import deepcopy
import numpy as np


class GlobalRandomForest:
    def __init__(self, max_depth=5, n_estimators=10, estimators_=None) -> None:
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.estimators_ = [] if estimators_ is None else estimators_

    def predict(self, X, soft=False, n_classes=-1):
        """Function to predict the class of the input data. It calculates the
        majority vote of the estimators.

        Args:
            X (ArrayLike): Array with the data to predict.

            (Below, new arguments in this implementation)
            soft (bool): Boolean indicating if soft predictions (probabilities) are returned. False by default
            n_classes (int): Number of classes in the data; if -1 (default), it is later calculated

        Returns:
            list: List with the predictions.
        """
        predictions = {}
        for estimator in self.estimators_:
            prediction = estimator.predict(X)
            for i, p in enumerate(prediction):
                if i not in predictions:
                    predictions[i] = []
                predictions[i].append(p)

        def most_common(lst):
            return max(set(lst), key=lst.count)

        def proba(lst, n_classes):
            probs = [None for _ in range(n_classes)]
            for c in range(n_classes):
                probs[c] = lst.count(c) / len(lst)
            return probs


        if soft:
            if n_classes <=0:
                n_classes = len(np.unique(np.concatenate([predictions[i] for i in range(len(predictions))])))
            return np.array([proba(predictions[i], n_classes) for i in range(len(predictions))])
        else:
            return [most_common(predictions[i]) for i in range(len(predictions))]


    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result