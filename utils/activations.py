import numpy as np

def sigmoid(x):
    # activation function
    return 1 / (1 + np.exp(-x))
    
def d_sigmoid(x):
    # derivative of activation for back prop
    return sigmoid(x) * (1 - sigmoid(x))