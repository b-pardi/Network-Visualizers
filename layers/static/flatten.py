import numpy as np
from layers.static.interfaces import Layer


# Flatten Layer
class FlattenLayer(Layer):
    def forward(self, inputs):
        """
        Reshape the multi-dimensional input into a 2D array.

        Args:
        - inputs: A 4D tensor of shape (batch_size, channels, height, width) or similar.

        Returns:
        - Flattened 2D tensor of shape (batch_size, channels * height * width).
        """
        self.in_shape = inputs.shape
        batch_size = self.in_shape[0]

        # keep batch_size and flatten all other dimensions.
        return inputs.reshape(batch_size, -1)

    def backward(self, d_out):
        """
        Reshape gradient output back to the input's original shape.
        Args:
        - d_out: A 2D tensor of shape (batch_size, flattened_size) representing the gradient from the next layer.
        
        Returns:
        - The gradient reshaped back to the original input shape (e.g., batch_size, channels, height, width).    
        """
        return d_out.reshape(self.in_shape)