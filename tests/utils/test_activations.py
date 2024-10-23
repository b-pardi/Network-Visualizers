import numpy as np
import pytest
from layers.static.activations import ReLULayer, SoftmaxLayer, TanhLayer, sigmoid, d_sigmoid

# Define a pytest fixture to set up the ReLU layer
@pytest.fixture
def relu_layer():
    """
    Fixture to create and return an instance of the ReLU layer.
    This fixture is used in multiple test functions.
    """
    return ReLULayer()

@pytest.fixture
def softmax_layer():
    """
    Fixture to create and return an instance of the Softmax layer.
    """
    return SoftmaxLayer()

@pytest.fixture
def tanh_layer():
    """
    Fixture to create and return an instance of the Tanh layer.
    """
    return TanhLayer()

# Define a pytest fixture for test inputs
@pytest.fixture
def test_inputs():
    """
    Fixture to provide a test input array for ReLU forward and backward tests.
    """
    return np.array([[1, -1, 0], [2, -2, 3]])

@pytest.fixture
def test_d_out():
    """
    Fixture to provide a test gradient (d_out) for backward pass tests.
    """
    return np.array([[1, 1, 1], [1, 1, 1]])

# Test the forward pass of the Sigmoid function using the shared inputs fixture
def test_sigmoid_forward(test_inputs):
    # Expected output after applying Sigmoid
    expected_output = np.array([[0.731058, 0.26894, 0.5], [0.880797, 0.119203, 0.952574]])
    
    # Run forward pass of sigmoid
    output = sigmoid(test_inputs)
    
    # Assert that the output is as expected
    assert np.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"

# Test the backward pass (derivative) of the Sigmoid function using the shared inputs fixture
def test_d_sigmoid(test_inputs):
    # Expected output of the sigmoid derivative
    expected_d_sigmoid = np.array([[0.196612, 0.196612, 0.25], [0.104993, 0.104993, 	0.045176]])
    
    # Run derivative of sigmoid
    d_output = d_sigmoid(test_inputs)
    
    # Assert that the derivative is as expected
    assert np.allclose(d_output, expected_d_sigmoid, atol=1e-3), f"Expected {expected_d_sigmoid}, but got {d_output}"

# Test the forward pass of the ReLU layer using the fixture
def test_relu_forward(relu_layer, test_inputs):
    # Expected output after applying ReLU
    expected_output = np.array([[1, 0, 0], [2, 0, 3]])
    
    # Run forward pass using the fixture-provided relu_layer and inputs
    output = relu_layer.forward(test_inputs)
    
    # Assert that the output is as expected
    assert np.array_equal(output, expected_output), f"Expected {expected_output}, but got {output}"

# Test the backward pass of the ReLU layer using the fixture
def test_relu_backward(relu_layer, test_inputs, test_d_out):
    # Perform forward pass (to store inputs in the layer)
    relu_layer.forward(test_inputs)
    
    # Expected gradient after backward pass
    expected_dX = np.array([[1, 0, 0], [1, 0, 1]])
    
    # Run backward pass
    dX = relu_layer.backward(test_d_out)
    
    # Assert that the gradients are as expected
    assert np.array_equal(dX, expected_dX), f"Expected {expected_dX}, but got {dX}"

# Test Softmax forward pass
def test_softmax_forward(softmax_layer, test_inputs):
    # Expected output after applying Softmax
    expected_output = np.array([
        [0.66524096, 0.09003057, 0.24472847],
        [0.26762315, 0.00490169, 0.72747516]
    ])
    
    # Run forward pass using the fixture-provided softmax_layer and inputs
    output = softmax_layer.forward(test_inputs)
    
    # Assert that the output is as expected
    assert np.allclose(output, expected_output, atol=1e-6), f"Expected {expected_output}, but got {output}"

# Test Softmax backward pass
def test_softmax_backward(softmax_layer, test_inputs, test_d_out):
    # Perform forward pass (to store softmax outputs in the layer)
    softmax_layer.forward(test_inputs)
    
    # Define expected gradient
    expected_dX = np.zeros((2, 3))

    # Run backward pass
    dX = softmax_layer.backward(test_d_out)
    
    # Assert that the gradients are as expected
    assert np.allclose(dX, expected_dX, atol=1e-6), f"Expected {expected_dX}, but got {dX}"

# Test Tanh forward pass
def test_tanh_forward(tanh_layer, test_inputs):
    # Expected output after applying Tanh
    expected_output = np.array([
        [0.76159416, -0.76159416, 0],
        [0.96402758, -0.96402758, 0.99505475]
    ])
    
    # Run forward pass using the fixture-provided tanh_layer and inputs
    output = tanh_layer.forward(test_inputs)
    
    # Assert that the output is as expected
    assert np.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"

# Test Tanh backward pass
def test_tanh_backward(tanh_layer, test_inputs, test_d_out):
    # Perform forward pass (to store tanh outputs in the layer)
    tanh_layer.forward(test_inputs)
    
    # Expected gradient after backward pass
    expected_dX = np.array([
        [0.41997434, 0.41997434, 1],
        [0.07065082, 0.07065082, 0.00986604]
    ])
    
    # Run backward pass
    dX = tanh_layer.backward(test_d_out)
    
    # Assert that the gradients are as expected
    assert np.allclose(dX, expected_dX), f"Expected {expected_dX}, but got {dX}"