import numpy as np

def max_norm(arr):
    return arr / np.max(arr)

def constant_norm(arr, c):
    return arr / c