import numpy as np
import pytest

from src.mnist_digits import MaxPoolingLayer, FlattenLayer, DenseLayer

@pytest.fixture
def pooling_layer():
    """
    Fixture to create and return an instance of MaxPoolingLayer with kernel size 2 and stride 2.
    """
    return MaxPoolingLayer(filter_size=2, stride=2)

@pytest.fixture
def test_input():
    """
    Fixture to provide a simple test input for the max pooling operation.
    """
    # Simple 1x1 channel 4x4 input for max pooling
    return np.array([[[[1, 3, 2, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]]]], dtype=np.float32)

def test_max_pooling_forward(pooling_layer, test_input):
    """
    Test the forward pass of max pooling using a basic 1x1 channel 4x4 input.
    """
    # Expected output after applying 2x2 max pooling with stride 2
    expected_output = np.array([[[[6, 8],
                                  [14, 16]]]], dtype=np.float32)
    
    # Perform the forward pass
    output = pooling_layer.forward(test_input)
    
    # Assert that the output matches the expected output
    assert np.array_equal(output, expected_output), f"Expected {expected_output}, but got {output}"

def test_max_pooling_backward(pooling_layer):
    """
    Test the backward pass of max pooling using a basic 4x4 input.
    """
    # Input and d_out for a 4x4 input with 2x2 pooling and stride 2
    inputs = np.array([[[[1, 3, 2, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]]]], dtype=np.float32)
    
    d_out = np.array([[[[1, 2],
                        [3, 4]]]], dtype=np.float32)
    
    pooling_layer.forward(inputs)  # Perform the forward pass to store input and output values
    dX = pooling_layer.backward(d_out)
    
    # Expected gradient output
    expected_dX = np.array([[[[0, 0, 0, 0],
                              [0, 1, 0, 2],
                              [0, 0, 0, 0],
                              [0, 3, 0, 4]]]], dtype=np.float32)
    
    # Check that the backward pass gives the correct gradients
    assert np.array_equal(dX, expected_dX), f"Expected {expected_dX}, but got {dX}"