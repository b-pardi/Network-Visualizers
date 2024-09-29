import numpy as np
import pytest
from utils.activations import ReLULayer, sigmoid, d_sigmoid

# Define a pytest fixture to set up the ReLU layer
@pytest.fixture
def relu_layer():
    """
    Fixture to create and return an instance of the ReLU layer.
    This fixture is used in multiple test functions.
    """
    return ReLULayer()

# Define a pytest fixture for test inputs
@pytest.fixture
def test_inputs():
    """
    Fixture to provide a test input array for ReLU forward and backward tests.
    """
    return np.array([[1, -1, 0], [2, -2, 3]])

# Test the forward pass of the Sigmoid function using the shared inputs fixture
def test_sigmoid_forward(test_inputs):
    # Expected output after applying Sigmoid
    expected_output = 1 / (1 + np.exp(-test_inputs))
    
    # Run forward pass of sigmoid
    output = sigmoid(test_inputs)
    
    # Assert that the output is as expected
    assert np.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"

# Test the backward pass (derivative) of the Sigmoid function using the shared inputs fixture
def test_d_sigmoid_backward(test_inputs):
    # Expected output of the sigmoid derivative
    sigmoid_output = sigmoid(test_inputs)
    expected_d_sigmoid = sigmoid_output * (1 - sigmoid_output)
    
    # Run derivative of sigmoid
    d_output = d_sigmoid(test_inputs)
    
    # Assert that the derivative is as expected
    assert np.allclose(d_output, expected_d_sigmoid), f"Expected {expected_d_sigmoid}, but got {d_output}"

# Test the forward pass of the ReLU layer using the fixture
def test_relu_forward(relu_layer, test_inputs):
    # Expected output after applying ReLU
    expected_output = np.array([[1, 0, 0], [2, 0, 3]])
    
    # Run forward pass using the fixture-provided relu_layer and inputs
    output = relu_layer.forward(test_inputs)
    
    # Assert that the output is as expected
    assert np.array_equal(output, expected_output), f"Expected {expected_output}, but got {output}"

# Test the backward pass of the ReLU layer using the fixture
def test_relu_backward(relu_layer, test_inputs):
    # Perform forward pass (to store inputs in the layer)
    relu_layer.forward(test_inputs)
    
    # Define gradient coming from the next layer (d_out)
    d_out = np.array([[1, 1, 1], [1, 1, 1]])
    
    # Expected gradient after backward pass
    expected_dX = np.array([[1, 0, 0], [1, 0, 1]])
    
    # Run backward pass
    dX = relu_layer.backward(d_out)
    
    # Assert that the gradients are as expected
    assert np.array_equal(dX, expected_dX), f"Expected {expected_dX}, but got {dX}"
