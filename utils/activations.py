import numpy as np

from utils.interfaces import Layer

def sigmoid(x):
    # activation function
    return 1 / (1 + np.exp(-x))
    
def d_sigmoid(x):
    # derivative of activation for back prop
    return sigmoid(x) * (1 - sigmoid(x))

# ReLU Activation Layer
class ReLULayer(Layer):
    """
    ReLU (Rectified Linear Unit) Activation Function:

    - ReLU is defined as f(x) = max(0, x), where it outputs the input if positive and 0 otherwise.
    - It introduces non-linearity to the model, enabling learning of complex patterns.
    - ReLU is computationally efficient and helps mitigate the vanishing gradient problem, making it ideal for deep networks.
    - Commonly used in CNNs for image tasks due to its ability to preserve key features and enable faster training.
    """
    def forward(self, inputs):
        """
        - Apply ReLU function: f(x) = max(0, x)

        Args:
        - inputs: The original input data to this layer during the forward pass.
                Shape: (batch_size, in_channels, in_size, in_size)
        """
        self.inputs = inputs
        return np.maximum(0, inputs) # use np.maximum instead of np.max to compare elementwise

    def backward(self, d_out):
        """
        Backward pass through the ReLU activation function.
        - Apply ReLU derivative:
        f`(x) = {
            1, if x > 0;
            0, if x<= 0;
        }
        
        Args:
        - d_out: Gradient of the loss with respect to the output of the ReLU.
        - inputs: The original inputs to this layer from the forward pass.
        
        Returns:
        - dX: Gradient of the loss with respect to the inputs (only where inputs > 0).
        """
        grad_mask = self.inputs > 0 # binary mask, fills with 1s cases where inputs are 0
        return d_out * grad_mask