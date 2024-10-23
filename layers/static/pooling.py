import numpy as np
from layers.static.interfaces import Layer


# Max Pooling Layer
class MaxPoolingLayer(Layer):
    """
    MaxPoolingLayer:

    The operation extracts the maximum value in each non-overlapping region (window) 
    defined by the kernel size and stride.
    """
    def __init__(self, filter_size=3, stride=1):
        # no weights or biases for pooling
        self.filter_size = filter_size
        self.stride = stride
    
    def forward(self, inputs):
        """
        Forward pass of the max pooling operation.
        It uses a pooling window (kernel) and a stride to slide over the input and extract
        the maximum value in each region.

        Unlike convolution we look at each filter independently instead of altogether
        
        Args:
        - inputs: 4D tensor of shape (batch_size, num_channels, height, width).

        Returns:
        - Pooled output tensor, reduced in height and width.
        """
        self.inputs = inputs
        batch_size, num_channels, in_size, _ = inputs.shape

        # get output size post pooling and init output array with that size
        self.out_size = (in_size - self.filter_size) // self.stride + 1
        self.pool_out = np.zeros((batch_size, num_channels, self.out_size, self.out_size), dtype=np.float32)

        for img_idx in range(batch_size):                       # iterate over all imgs in batch
            for filter_idx in range(num_channels):              # iterate over each filter (output channel)
                for y in range(0, in_size - self.filter_size + 1, self.stride):        # iterate over input map vertically, incrementing by stride
                    for x in range(0, in_size - self.filter_size + 1, self.stride):    # iterate over input map horizontally, incrementing by stride
                        # get section of input from pool window
                        input_patch = inputs[img_idx, filter_idx, y:y + self.filter_size, x:x + self.filter_size]

                        # find max value of window across each individual filter
                        max_val = np.max(input_patch)
                        self.pool_out[img_idx, filter_idx, y // self.stride, x // self.stride] = max_val

        return self.pool_out

    def backward(self, d_out):
        """
        Backward pass of the max pooling layer.

        The backward pass for max pooling computes the gradient of the loss with respect to the input of the pooling layer.
        In max pooling, during the forward pass, only the maximum value in each pooling window is propagated forward.
        Therefore, in the backward pass, the gradient from the next layer (d_out) is passed only to the position of the 
        maximum value within the original input patch, and the other positions within the patch receive zero gradients.

        Args:
        - d_out (ndarray): The gradient of the loss with respect to the output of the max pooling layer. It has the same shape
        as the pooled output, which is (batch_size, num_channels, out_height, out_width).

        Returns:
        - dX (ndarray): The gradient of the loss with respect to the input of the max pooling layer (same shape as the original input),
        with shape (batch_size, num_channels, in_height, in_width). The gradients are propagated only to the positions 
        of the maximum values that were selected during the forward pass, while all other positions receive zero gradients.
        """
        # extract dimensions of inputs and d_out (gradients)
        batch_size, num_channels, in_size, _ = self.inputs.shape

        # gradient array
        dX = np.zeros_like(self.inputs)

        for img_idx in range(batch_size):                       # iterate over all imgs in batch
            for filter_idx in range(num_channels):              # iterate over each filter (output channel)
                for y in range(0, in_size - self.filter_size + 1, self.stride):        # iterate over input map vertically, incrementing by stride
                    for x in range(0, in_size - self.filter_size + 1, self.stride):    # iterate over input map horizontally, incrementing by stride
                        # get section of input from pool window
                        input_patch = self.inputs[img_idx, filter_idx, y:y + self.filter_size, x:x + self.filter_size]

                        # find location of max value
                        max_val = np.max(input_patch)
                        mask = (input_patch == max_val)

                        # incoming gradient from d_out is pooled only to the location of the max value
                        # index for d_out is [y // self.stride, x // self.stride] in the pooled output
                        
                        #print(batch_size, filter_idx, type(img_idx), type(filter_idx), type(y // self.stride), type(x // self.stride), type(mask), type(d_out))
                        #print(d_out, '\n', d_out.shape, type(d_out), type(d_out[0]))
                        #print(d_out.shape)
                        max_grad = mask * d_out[img_idx, filter_idx, y // self.stride, x // self.stride]
                        dX[img_idx, filter_idx, y:y + self.filter_size, x:x + self.filter_size] += max_grad
        return dX
        