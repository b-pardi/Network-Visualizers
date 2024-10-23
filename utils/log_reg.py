import numpy as np
from layers.static.activations import sigmoid

class LogisticRegression:
    def __init__(self, lr=0.01, iters=8000):
        self.lr = lr
        self.iters = iters
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        y = y.reshape(-1) # proper 1d array
        m = len(y)

        for i in range(self.iters):
            yhat = self.forward(X)

            # derivative of BCE loss fn wrt to output logit (a or sigmoid(z)) ends up being yhat - ytrue
            errs = yhat - y
            
            # dL/dw = dL/dz * dz/dw where z = w.Tx + b so dL/dw = (yhat - y)X
            dw = np.dot(X.T, errs) / m # dw = (1/m) * sum((hat{y} - y) * x)
            # dL/db = yhat - y; since dz/db = 1
            db = np.sum(errs) / m # db = (1/m) * sum(hat{y} - y)

            # update weights incrementally in direction of gradient
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def forward(self, X):
        # logit computed as a linear combination of the input features z = w.Tx + b
        # w.Tx = sum_1:n(w_i*x_i)
        # essentially this line does w.T dot x_i for x_i in X
        z = np.dot(X, self.weights) + self.bias # input sum (nxn)
        a = sigmoid(z) # activated output
        return a