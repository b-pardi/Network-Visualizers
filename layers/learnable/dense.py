import numpy as np
from layers.static.interfaces import LearnableLayer
from utils.initializers import xavier_glorot_normal_init


# Fully Connected Layer (Dense)
class DenseLayer(LearnableLayer):
    """
    Dense Layer (Fully Connected Layer) performing an affine transformation on the input: y = xW + b
    
    The dense layer is a fully connected layer, meaning each input node is connected to every output node via learnable 
    weights. It is used to combine features learned by convolutional or other layers into a more compact representation 
    and is often used as the final layer in a network for tasks such as classification.

    Attributes:
    - in_features (int): Number of input features (i.e., the size of the input vector).
    - out_features (int): Number of output features (i.e., the number of neurons in the layer).
    - weights (ndarray): The learnable weight matrix of shape (in_features, out_features).
    - biases (ndarray): The learnable bias vector of shape (out_features,).
    
    The layer performs a matrix multiplication of the input with the weights and adds the biases:
    Output = X (dot) W + b, 
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = xavier_glorot_normal_init(in_features, out_features, (in_features, out_features))
        self.biases = np.zeros(out_features, dtype=np.float32)

    def forward(self, inputs):
        """
        Forward pass of the dense layer.
        y = xW + b

        y: (batch_size x out_features)
        x: (batch_size x in_features)
        W: (in_features x out_features)
        b: (1 x out_features)
        
        Args:
        - inputs (ndarray): A 2D tensor of shape (batch_size, in_features) where each row represents an input vector.
        
        Returns:
        - output (ndarray): A 2D tensor of shape (batch_size, out_features) where each row represents the output 
          after the matrix multiplication and bias addition.
        """
        self.inputs = inputs
        #print(inputs.shape, inputs[0], '\n', self.weights.shape, self.weights[0])
        self.outputs = np.dot(inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, d_out):
        """
        Backward pass of the dense layer (backpropagation).
        find gradients of loss w.r.t inputs, weights, biases

        (d represents a partial here, and del is delta, shorthand for the partial of the var w.r.t loss fn)
        given that the gradient of the loss w.r.t outputs is dL/dy = del_y,
        and we know that y = XW + b,
        and since loss L depends on y, which depends on each of inputs, weights, and biases
        inputs - dL/dX = dL/dy * dy/dx -> del_x = del_y @ W.T
        weights - dL/dW = dL/dy * dy/dW -> del_W = x.T @ del_y
        biases - dL/db = dL/dy * dy/db - > del_b = 

        Note: matrices become transposed as needed after derivative to align dimensions
        Basically when derivating matrices, formally it is done in summation notation,
        then we move/transpose matrices to fit the matrix multiplication definition:
        C_ij = sum_k:n[ A_ik @ B_kj ]
        showing captured summations over the shared index k.

        Args:
        - d_out (ndarray): Gradient of the loss with respect to the output of the dense layer (same shape as the output).
        
        Returns:
        - d_inputs (ndarray): Gradient of the loss with respect to the input of the dense layer.
        - d_weights (ndarray): Gradient of the loss with respect to the weights of the dense layer.
        - d_biases (ndarray): Gradient of the loss with respect to the biases of the dense layer.
        """
        # Gradient w.r.t input x: delta_x = delta_y @ W.T
        # Using chain rule: delta_x = delta_y * dy/dx
        # Since y = xW + b, dy/dx = W
        delta_x = np.dot(d_out, self.weights.T)
        # delta_x shape: (N, D_in)

        # Gradient w.r.t weights W: delta_W = x.T @ delta_y
        # Using chain rule: delta_W = x^T * delta_y
        # Each element of delta_W is the sum over the batch of input features times output gradients
        delta_W = np.dot(self.inputs.T, d_out)
        # delta_W shape: (D_in, D_out)

        # Gradient w.r.t biases b: delta_b = sum over batch of delta_y
        # Since each b_k affects each y_k equally, we sum over the batch
        delta_b = np.sum(d_out, axis=0)
        # delta_b shape: (D_out,)

        # Store gradients for parameter updates
        self.grad_W = delta_W
        self.grad_b = delta_b

        return delta_x  # Pass gradient back to previous layer
