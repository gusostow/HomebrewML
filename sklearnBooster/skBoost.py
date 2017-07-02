import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

#used to keep weights summing to one
softmax = lambda x: np.exp(x) / np.sum(np.exp(x))

class BoostedClassifier(object):

    def __init__(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = [clone(self.base_estimator) for _ in range(self.n_estimators)]

    def fit(self, X, y):

        pt_weights = np.array([1. / X.shape[0] for _ in range(X.shape[0])])
        estimator_weights = np.ones(self.n_estimators)

        pred = []
        for indx, estimator in enumerate(self.estimators):

            train_ind = np.array(range(X.shape[0]))
            train_ind_weighted = np.random.choice(train_ind,\
                                                  X.shape[0], p = pt_weights)

            X_weighted = X[train_ind_weighted]
            y_weighted = y[train_ind_weighted]

            estimator.fit(X_weighted, y_weighted)
            pred.append(estimator.predict(X))

            wrong_points = estimator.predict(X_weighted) != y_weighted
            wrong_points = wrong_points.astype("float")

            loss = np.dot(wrong_points, pt_weights) / np.sum(pt_weights)

            if not loss: #prevent divide by zero
                break

            estimator_weight = np.log((1-loss) / loss)
            estimator_weights[indx] = estimator_weight

            pt_weights *= np.exp(1 * estimator_weight * wrong_points)
            pt_weights = softmax(pt_weights)

        self.estimator_weights = estimator_weights
        self._pt_weights_ = pt_weights
        self.pred = np.column_stack(pred)

    def predict(self, X):

        pred_matrix = np.column_stack([estimator.predict(X) for estimator in self.estimators])
        pred_matrix[pred_matrix == 0] = -1

        raw_predictions = pred_matrix.dot(self.estimator_weights)
        predictions = np.where(raw_predictions >= 0, 1, 0)

        return predictions
