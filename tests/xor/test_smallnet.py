import pytest
import numpy as np
from src.xor import SmallNet

@pytest.fixture
def xor_data():
    # XOR inputs and expected outputs
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    return X, y

@pytest.fixture
def small_net():
    # Initialize the SmallNet with 3 hidden neurons
    return SmallNet(h=3, epochs=10000, num_epochs_per_refresh=1000)

def test_forward(small_net, xor_data):
    X, _ = xor_data
    # Forward pass
    output = small_net.forward(X)

    # Check the shape of the output
    assert output.shape == (4, 1), f"Expected output shape (4, 1), but got {output.shape}"

    # Check that output values are in the range (0, 1)
    assert np.all(output >= 0) and np.all(output <= 1), "Outputs are not in the range (0, 1)"

def test_backward(small_net, xor_data):
    X, y = xor_data
    initial_w1 = small_net.w1.copy()
    initial_b1 = small_net.b1.copy()
    initial_w2 = small_net.w2.copy()
    initial_b2 = small_net.b2.copy()

    # Perform one forward and backward pass
    small_net.forward(X)
    small_net.backward(X, y, lr=0.1)

    # Check if weights and biases have been updated
    assert not np.array_equal(initial_w1, small_net.w1), "Weights w1 did not update"
    assert not np.array_equal(initial_b1, small_net.b1), "Biases b1 did not update"
    assert not np.array_equal(initial_w2, small_net.w2), "Weights w2 did not update"
    assert not np.array_equal(initial_b2, small_net.b2), "Biases b2 did not update"

    # Check if the updates are not too large
    assert np.all(np.abs(small_net.w1 - initial_w1) < 1), "Weight update w1 too large"
    assert np.all(np.abs(small_net.b1 - initial_b1) < 1), "Bias update b1 too large"
    assert np.all(np.abs(small_net.w2 - initial_w2) < 1), "Weight update w2 too large"
    assert np.all(np.abs(small_net.b2 - initial_b2) < 1), "Bias update b2 too large"
