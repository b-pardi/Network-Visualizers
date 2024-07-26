import numpy as np
import requests
import gzip
import os

from utils.dl_data import download_mnist

# Enforce an interface to ensure all derived classes have forward and backward methods
class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError
    
class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        self.in_channels = in_channels # channels of input data, init 3 for color (rgb) or 1 for grayscale
        self.out_channels = out_channels # represents number of kernels in the conv layer
        self.kernel_size = kernel_size # dimensions of filter to pass over images (square)
        self.stride = stride # how many pixels we step when scanning img with kernel
        self.pad = pad # how many empty pixels we 'pad' the border with
        
        self.weights = np.random.randn(kernel_size, kernel_size, in_channels, out_channels) * 0.1
        self.biases = np.zeros((out_channels,)) # one bias term for each filter (out channel)

def run():
    download_mnist("res/mnist_data/")