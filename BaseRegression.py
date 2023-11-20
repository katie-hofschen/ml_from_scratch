import numpy as np
from activation_functions import sigmoid


class BaseRegression:
    def __init__(self, lr=0.001, n_iter=100):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        if not isinstance(lr, float) or lr < 0.0:
            raise ValueError("The learning rate should be a number above 0.")
        self._lr = lr

    @property
    def n_iter(self):
        return self._n_iter

    @n_iter.setter
    def n_iter(self, n_iter):
        if not isinstance(n_iter, int) or n_iter < 0:
            raise ValueError("The number of iterations should be a number above 0.")
        self._n_iter = n_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # TODO use different intialization methods eg. Xavier init., ...
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weights) + self.bias
            errors = y_pred - y

            db = (1 / n_samples) * np.sum(errors)
            dw = (1 / n_samples) * np.dot(X.T, errors)

            self.bias -= self.lr * db
            self.weights -= self.lr * dw

    def _approximation(self, X, w, b):
        raise NotImplementedError()

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def _predict(self, X, w, b):
        raise NotImplementedError()
