import numpy as np
import pytest

from src.mnist_digits import Conv2D

def test_conv_forward():
    """
    Test the forward pass of the convolutional layer with a simple example.
    """
    # Create an instance of the ConvLayer with known parameters
    conv_layer = Conv2D( # default conv layer args
        in_channels=1,  
        out_channels=1,
        filter_size=3, 
        stride=1,        
        pad=0             
    )

    # Manually set the weights and bias for predictability
    conv_layer.weights = np.ones((1, 1, 3, 3))  # setting filter weights to all ones
    conv_layer.bias = np.zeros((1,))            # setting bias to zero

    # simple test input tensor with known values
    # (batch_size, in_channels, height, width)
    test_input = np.array([[[[1, 2, 3, 0],
                             [4, 5, 6, 0],
                             [7, 8, 9, 0],
                             [0, 0, 0, 0]]]])  # Shape: (1, 1, 4, 4)

    # convolve
    output = conv_layer.forward(test_input)

    # the expected output is calculated manually based on the input and filter weights
    expected_output = np.array([[[[45, 33],
                                  [39, 28]]]])  # Shape: (1, 1, 2, 2)

    # Check that the output from the forward pass matches the expected output
    assert np.array_equal(output, expected_output), f"Expected {expected_output}, but got {output}"
