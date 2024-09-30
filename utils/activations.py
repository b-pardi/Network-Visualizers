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
    def forward(self, x):
        """
        Apply ReLU function: f(x) = max(0, x)

        Args:
            inputs: The original input data to this layer during the forward pass.
                Shape: (batch_size, in_channels, in_size, in_size)
        """
        self.x = x
        return np.maximum(0, x) # use np.maximum instead of np.max to compare elementwise

    def backward(self, d_out):
        """
        Backward pass through the ReLU activation function.
        - Apply ReLU derivative:
        f`(x) = {
            1, if x > 0;
            0, if x<= 0;
        }
        
        Args:
            d_out: Gradient of the loss with respect to the output of the ReLU.
            inputs: The original inputs to this layer from the forward pass.
        
        Returns:
            dX: Gradient of the loss with respect to the inputs (only where inputs > 0).
        """
        grad_mask = self.x > 0 # binary mask, fills with 1s cases where inputs are 0
        return d_out * grad_mask
    
class SoftmaxLayer(Layer):
    """
    SoftmaxLayer:
    
    The Softmax layer is commonly used in the output layer of a classification neural network. 
    It transforms raw logits (unnormalized output scores) into a probability distribution over 
    multiple classes. The softmax function ensures that the sum of the probabilities over 
    all classes equals 1, making it suitable for multi-class classification tasks.
    
    Formula:
    - The softmax function for the i-th class is given by:
      softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j in the classes)
    
    Usage:
    - Softmax is typically used in the output layer of neural networks when the goal is to 
      predict a class label. It is often paired with a loss function like categorical 
      cross-entropy to compute the loss between the predicted probability distribution 
      and the true class labels.
    
    Methods:
    - forward: Applies the softmax function to the input (logits).
    - backward: Computes the gradient of the loss with respect to the input (logits) 
      using the softmax Jacobian matrix.
    """
    def forward(self, x):
        """
        Compute the softmax of each element along the last axis of the input array.

        Args:
        - x: Input array, typically logits. Can be of shape (batch_size, num_classes).

        Returns:
        - Softmax probabilities with the same shape as the input, where each value is normalized 
          in the range (0, 1), and the values along the last axis sum to 1.
        """        
        # apply to last axis since softmax is applied on output layers containing logits
        shifted = x - np.max(x, axis=-1, keepdims=True) # shifting helps prevent overflow without effecting outcome
        exps = np.exp(shifted)
        self.sm_out = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.sm_out
    
    def backward(self, d_out):
        """Compute the gradient of the loss with respect to the input of the softmax layer.
        Args:
            d_out: Gradient of the loss with respect to the output of the softmax layer. 
                 This is usually passed in from the next layer during backpropagation.

        Returns:
            dX: Gradient of the loss with respect to the input of the softmax layer (same shape as input).
        """
        sm_out = self.sm_out
        
        # Compute the dot product between the softmax output and the upstream gradient
        sum_d_out_sm = np.sum(d_out * sm_out, axis=1, keepdims=True)

        dX = dX = sm_out * (d_out - sum_d_out_sm)
        return dX

class TanhLayer(Layer):
    """
    TanhLayer:
    
    The Tanh (hyperbolic tangent) layer applies the tanh activation function to the input. 
    The tanh function maps input values to the range (-1, 1) and is often used in hidden layers 
    of neural networks to introduce non-linearity.
    
    Formula:
    - The tanh function for input x is given by:
      tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Usage:
    - Tanh is commonly used in hidden layers of neural networks, especially when the model 
      benefits from output values that are centered around zero.
    
    Methods:
    - forward: Applies the tanh function to the input.
    - backward: Computes the gradient of the loss with respect to the input using the derivative of tanh.
    """
    def forward(self, x):
        self.inputs = x
        self.tanh_output = np.tanh(x)  # Apply the tanh function element-wise
        return self.tanh_output
    
    def backward(self, d_out):
        ''' d(tanh(x)) = 1 - tanh(x)^2 '''
        return d_out * (1 - self.tanh_output**2)