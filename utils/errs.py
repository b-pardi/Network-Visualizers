import numpy as np

def mse(y, y_hat):
    return np.mean(np.square(y - y_hat))