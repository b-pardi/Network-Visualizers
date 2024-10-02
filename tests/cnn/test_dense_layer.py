import numpy as np
import pytest

from src.mnist_digits import DenseLayer

@pytest.fixture
def dense_layer():
    """
    Fixture to create instance of FlattenLayer
    """
    dl = DenseLayer(16, 4)
    dl.weights = np.array([[0.1] * 4] * 16, dtype=np.float32)  # All weights set to 0.1 for simplicity
    dl.biases = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)  # Simple bias values

    return dl

@pytest.fixture
def test_input():
    """
    Fixture to provide a simple test input for the max pooling operation.
    """
    # Simple 1x1 channel 4x4 input for max pooling
    return np.array([[[[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]]]], dtype=np.float32)

@pytest.fixture
def test_d_out():
    return np.array([[[[1, 2],
                       [3, 4]]]], dtype=np.float32)

def test_dense_forward(dense_layer, test_input):
    test_input = test_input.reshape(1, -1)
    output = dense_layer.forward(test_input)
    
    # Manually calculate the expected output:
    # y = X @ W + b
    expected_output = np.array([[13.7, 13.8, 13.9, 14. ]], dtype=np.float32)
    
    # Assert that the output matches the expected output
    assert np.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"

def test_dense_backward(dense_layer, test_input, test_d_out):
    """
    Test the backward pass of the DenseLayer using manually calculated expected values.
    """
    test_input = test_input.reshape(1, -1)
    test_d_out = test_d_out.reshape(1, -1)

    dense_layer.forward(test_input)
    d_inputs = dense_layer.backward(test_d_out)

    # Expected gradient w.r.t. inputs (dX):
    expected_dX = np.ones((1, 16), dtype=np.float32)  # (1, 16)

    # Expected gradient w.r.t. weights (dW):
    expected_dW = np.array([
        [1, 2, 3, 4],
        [2, 4, 6, 8],
        [3, 6, 9, 12],
        [4, 8, 12, 16],
        [5, 10, 15, 20],
        [6, 12, 18, 24],
        [7, 14, 21, 28],
        [8, 16, 24, 32],
        [9, 18, 27, 36],
        [10, 20, 30, 40],
        [11, 22, 33, 44],
        [12, 24, 36, 48],
        [13, 26, 39, 52],
        [14, 28, 42, 56],
        [15, 30, 45, 60],
        [16, 32, 48, 64]])  # (16, 4)
    
    # Expected gradient w.r.t. biases (db):
    expected_db = np.array([[1,2,3,4]])  # (4,)

    # Assert gradients are correct
    assert np.allclose(dense_layer.grad_W, expected_dW), f"Expected dW {expected_dW}, but got {dense_layer.grad_W}"
    assert np.allclose(dense_layer.grad_b, expected_db), f"Expected db {expected_db}, but got {dense_layer.grad_b}"
    assert np.allclose(d_inputs, expected_dX), f"Expected dX {expected_dX}, but got {d_inputs}"