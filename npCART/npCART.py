import numpy as np
import pandas as pd

def entropy(y):
    """
    Compute entropy of the outcome variable
    """
    total = y.size
    value_counts = np.bincount(y).astype("float")
    proportions = value_counts / y.size

    return sum(-i * np.log(i) for i in proportions if i)

def gain(X, y, column):
    """
    Compute information gain of a partition
    """
    prior_entropy = entropy(y)
    total = y.size

    values = X[column].unique()
    proportions = X[column].value_counts() / total
    return prior_entropy - sum(proportions[i] * \
               entropy(y[np.array(X[column]) == i]) for i in values)

def classify(series, tree):
    """
    Classify Pandas series with an existing decision tree
    """
    feature = tree[0]
    subtree = tree[1]

    answer = series[feature]
    response = subtree[answer]

    if type(response) != list: #base case
        return subtree[answer]
    else:
        return classify(series, response) #recursive case

class CategoricalClassifier(object):
    """
    Multi-class decision tree classifier. Accepts only
    categorical variables.
    """
    def __init__(self, max_depth = None):
        self.max_depth = max_depth
        self.fitted = False

    def Classify(self, X):
        if not self.fitted:
            raise Exception("Warning: Tree has not yet been grown on training data")

        yhat = X.apply(lambda x: classify(x, self.tree), axis = 1)
        return yhat

    def grow_tree_(self, X, y, candidates = None):
        """
        Internal function to grow tree on training data. To fit use method GrowTree instead.
        """
        if not candidates: # all columns are candidates2
            candidates = X.columns.tolist()
        #should this be a leaf node?
        unique = np.unique(y)
        counts = np.bincount(y)
        nonzero_ind = np.nonzero(counts)[0]

        if nonzero_ind.size == 1:
            return unique[0] #base case: return pure outcome
        elif len(candidates) == 1:
            return unique[np.argmax(counts)] #base case: return mode outcome

        #choose a feature to split on
        gains = [candidate, gain(candidate) for candidate in candidates]
        split_feature = sorted(gains,  # choose feature with largest info gain
                               key = lambda x: x[1],
                               reverse = True)[0][0]

        #prepare new candidates for recursive call
        new_candidates = [i for i in candidates if i != split_feature]
        levels = X[split_feature].unique().tolist()

        subtree = {}

        for level in levels:

            X_with_level = X.loc[X[split_feature] == level, :]
            y_with_level = y[np.array(X[split_feature] == level)]
            subtree[level] = GrowTree(X_with_level,
                                       y_with_level,
                                       new_candidates) #recursive call

        return [split_feature, subtree]

    def GrowTree(self, X, y, candidates = None):
        self.tree = grow_tree_(X, y)

