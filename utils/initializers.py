import numpy as np

def xavier_glorot_normal_init(in_features, out_features, size):
    """
    Normal Distribution Xavier/Glorot initialization for weights in a neural network layer.
    Weights are sampled from N(0, sqrt(2 / (n_in + n_out)))
    
    Args:
    - in_features (int): Number of input features to the layer (e.g., number of neurons from the previous layer).
    - out_features (int): Number of output features from the layer (e.g., number of neurons in the current layer).
    - size (tuple of ints): multi dimension sizing of the weights array to be initialized to
    
    Returns:
    - ndarray: A weight matrix initialized with Xavier Glorot nomral distribution method of shape 'size'.
    """
    # normal xavier/glorot initialization to help maintain stability of activation variance. Likely overkill for mnist digits, but it's for pedagogical purposes
    stddev = np.sqrt(2 / (in_features + out_features))

    return np.random.normal(0, stddev, size=size).astype(np.float32)

def xavier_glorot_uniform_init(in_featrues, out_features, size):
    """
    Uniform Distribution Xavier/Glorot initialization for weights in a neural network layer.
    Weights are sampled from U(-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out))).
    
    Args:
    - in_features (int): Number of input features to the layer (e.g., number of neurons from the previous layer).
    - out_features (int): Number of output features from the layer (e.g., number of neurons in the current layer).
    - size (tuple of ints): multi dimension sizing of the weights array to be initialized to
    
    Returns:
    - ndarray: A weight matrix initialized with Xavier Glorot uniform distribution method of shape 'size'.
    """
    limit = np.sqrt(6 / (in_featrues + out_features))
    
    return np.random.uniform(-limit, limit, size=(size)).astype(np.float32)