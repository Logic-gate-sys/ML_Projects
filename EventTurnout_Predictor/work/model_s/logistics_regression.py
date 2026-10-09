import numpy as np


# 5 Logistic Regression (Classification)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            prediction = sigmoid(linear_pred)
            dw = (1/m) * np.dot(X.T, (prediction - y))
            db = (1/m) * np.sum(prediction - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        prediction = sigmoid(linear_pred)
        class_preds = [1 if i > 0.5 else 0 for i in prediction]
        return np.array(class_preds)