import numpy as np

def softmax(matrix):
    _softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
    return np.apply_along_axis(_softmax, 1, matrix)

class SoftmaxClassifier(object):

    def __init__(self):
        pass

    def _forward(self, X, W):
        return softmax(np.dot(X, W))

    def _get_grad(self, X, y, W):
        prob = self._forward(X, W)
        dprob = prob
        dprob[range(len(dprob)), y] -= 1
        dW = np.dot(X.T, dprob)
        return dW

    def fit(self, X, y):
        num_classes = np.unique(y).size
        num_cols = X.shape[1] + 1
        W = np.random.randn(num_cols, num_classes)
        X = np.column_stack((np.ones(X.shape[0]), X)) # bias column

        for _ in range(1000):
            grad = self._get_grad(X, y, W)
            W -= 1e-1 * grad

        self.W = W
        predictions = self._forward(X, W)
        correct_probs = predictions[range(len(predictions)), y]
        loss = np.sum(np.log(correct_probs))
        print "loss: ", loss

    def predict(self, X):
        prob = self._forward
        predictions = np.argmax(X, axis = 1)
        return predictions
