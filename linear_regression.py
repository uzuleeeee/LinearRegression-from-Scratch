import numpy as np


class LinearRegression:
    # constructor 
    def __init__(self, lr=0.001, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        # need one weight for each feature
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iterations):
            # calculate prediction
            y_predicted = np.dot(X, self.weights) + self.bias
            # dJ/dw
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # dJ/db
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
