import numpy as np
import pytest

from src.mnist_digits import Conv2D

@pytest.fixture
def conv_layer():
    conv = Conv2D(
        in_channels=1,
        out_channels=1,
        filter_size=3,
        stride=1,
        pad=0
    )
    # Manually set the weights and bias for predictability
    conv.weights = np.ones((1, 1, 3, 3))  # Setting filter weights to all ones
    conv.bias = np.zeros((1,))            # Setting bias to zero
    return conv

@pytest.fixture
def test_input():
    # (batch_size, in_channels, height, width)
    return np.array([[[[1, 2, 3, 0],
                       [4, 5, 6, 0],
                       [7, 8, 9, 0],
                       [0, 0, 0, 0]]]])  # Shape: (1, 1, 4, 4)

@pytest.fixture
def expected_forward_output():
    return np.array([[[[45, 33],
                       [39, 28]]]])  # Shape: (1, 1, 2, 2)

@pytest.fixture
def d_out():
    # Gradient of the loss with respect to the output (same shape as the output)
    return np.array([[[[1, 1],
                       [1, 1]]]], dtype=np.float32)  # Shape: (1, 1, 2, 2)

@pytest.fixture
def expected_dX():
    # Expected gradient with respect to the input (shape of the input)
    return np.array([[[[1, 2, 2, 1],
                       [2, 4, 4, 2],
                       [2, 4, 4, 2],
                       [1, 2, 2, 1]]]], dtype=np.float32)  # Shape: (1, 1, 4, 4)

@pytest.fixture
def expected_dW():
    # Expected gradient with respect to the weights (same shape as the weights)
    return np.array([[[[12, 16, 9],
                       [24, 28, 15],
                       [15, 17, 9]]]], dtype=np.float32)  # Shape: (1, 1, 3, 3)

@pytest.fixture
def expected_db():
    # Expected gradient with respect to the bias (shape of the bias)
    return np.array([4.0], dtype=np.float32)  # Sum of d_out (4 ones)

def test_conv_forward(conv_layer, test_input, expected_forward_output):
    # Perform the forward pass
    output = conv_layer.forward(test_input)
    
    # Check that the output matches the expected output
    assert np.array_equal(output, expected_forward_output), f"Expected {expected_forward_output}, but got {output}"

def test_conv_backward(conv_layer, test_input, d_out, expected_dX, expected_dW, expected_db):
    # Forward pass to initialize inputs
    conv_layer.forward(test_input)
    
    # Perform the backward pass
    dX = conv_layer.backward(d_out)

    # Check that the gradients match the expected values
    np.testing.assert_array_equal(dX, expected_dX)
    np.testing.assert_array_equal(conv_layer.grad_W, expected_dW)
    np.testing.assert_array_equal(conv_layer.grad_b, expected_db)