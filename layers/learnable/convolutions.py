import numpy as np
from layers.static.interfaces import LearnableLayer
from utils.initializers import xavier_glorot_normal_init

class Conv2D(LearnableLayer):
    """
    This class implements a 2D convolutional layer, which applies multiple learnable filters (or kernels) to the input image, 
    scanning it across the spatial dimensions, and creating feature maps that capture various aspects of the input, 
    such as edges, textures, or more complex patterns at deeper layers.

    These filters that slide over the images perform element-wise multiplication with input at each location and sums the results, adding filter's bias at the end.
    The result of this is a feature map of the original image, resembling the image less and less with subsequent conv layers
    
    Parameters:
    - in_channels (int): Number of input channels (e.g., 1 for grayscale images, 3 for RGB images).
    - out_channels (int): Number of filters (kernels) applied during the convolution, representing the number of output feature maps.
    - filter_size (int): Size of the square filter (kernel) applied to the input (e.g., 3x3, 5x5).
    - stride (int): The step size of the filter as it moves across the input.
    - pad (int): Number of zeros added to the border of the input for padding.

    Initialization:
    - Weights are initialized using a Normal Xavier/Glorot initialization,
    to help maintain the variance of the activations and gradients during training ensuring stable learning.
    - Biases are initialized to zero
    """
    def __init__(self, in_channels=1, out_channels=8, filter_size=3, stride=1, pad=0):
        self.in_channels = in_channels # channels of input data, init 3 for color (rgb) or 1 for grayscale
        self.out_channels = out_channels # represents number of kernels/filters in the conv layer
        self.filter_size = filter_size # dimensions of filter to pass over images (square)
        self.stride = stride # how many pixels we step when scanning img with kernel
        self.pad = pad # how many empty pixels we 'pad' the border with
        
        # normal xavier/glorot initialization to help maintain stability of activation variance. Likely overkill for mnist digits, but it's for pedagogical purposes
        # size ordered as such allows for easier mat mul (matches input dimensions)
        self.weights = xavier_glorot_normal_init(in_channels * filter_size **2, out_channels, (out_channels, in_channels, filter_size, filter_size))
        self.biases = np.zeros((out_channels,), dtype=np.float32) # one bias term for each filter (out channel)

    def zero_pad(self, inputs):
        """
        adds padding of 0 values around border of image depending on self.pad
        """
        padded_arr = np.pad(
            inputs,
            pad_width=((0,0),   # specify which of 4 dims to pad (only pad img, not batch size or in chans)
                       (0,0),
                       (self.pad, self.pad),
                       (self.pad, self.pad)),
            mode='constant',    # pads array with value defined in constant_values arg
            constant_values=0   # num to pad img with
        )
        
        return padded_arr

    def forward(self, inputs):
        """
        Performs the forward pass of the 2D convolution operation. The method takes the input image(s) 
        and applies a set of filters (kernels) to produce output feature maps.

        Args:
        - inputs (ndarray): A 4D input tensor of shape (batch_size, in_channels, in_size, in_size), 
          where `batch_size` is the number of images in the batch, `in_channels` is the number of channels 
          (e.g., 1 for grayscale, 3 for RGB), and size are the spatial dimensions of the input (square images only).

        Returns:
        - out_feature_map (ndarray): A 4D tensor representing the output feature maps after applying convolution, 
          with shape (batch_size, out_channels, out_size, out_size). Each output channel corresponds to one filter, 
          and out_size is determined by the input size, filter size, stride, and padding.

        Process:
        1. Zero Padding (if applicable): If padding is enabled (`self.pad > 0`), the input is padded 
           to control the output size.
        2. Convolution: For each input image and each filter:
            - The filter is applied to a section of the input, performing an element-wise 
              multiplication with the patch and summing the results.
            - The filter then moves across the image, stepping by the stride value, and this process is repeated 
              for the entire image.
        3. Bias Addition: After convolution, a bias term is added to the result.
        4. Output Feature Map: The results from all filters are stored in the output feature map, 
           which represents the convolved output.
        
        The result is a compressed representation of the input that emphasizes certain features (e.g., edges, 
        textures) learned by the filters, making it suitable for further processing in deeper network layers.
        """
        # receives 4D array (batch_size, in_channels, in_size, in_size) -> [64, 1, 28, 28]
        # zero pad image if self.pad is not 0
        if self.pad > 0:
            inputs = self.zero_pad(inputs)

        self.inputs = inputs

        in_size = inputs.shape[-1] # dimension of square image input
        batch_size = inputs.shape[0] # number of images in batch

        # size of "img" post convolution (dimension of output after passing filter)
        out_size = (in_size - self.filter_size) // self.stride + 1

        # init output of feature map which stores conv results
        out_feature_map = np.zeros((batch_size, self.out_channels, out_size, out_size), dtype=np.float32)

        # convolution process
        for img_idx in range(batch_size):           # iterate over all imgs in batch
            for filter in range(self.out_channels): # iterate over each filter (output channel)
                for y in range(out_size):           # iterate over map vertically
                    for x in range(out_size):       # iterate over map horizontally
                        # get start and end indices of where the filter currently is in the img
                        filter_start_x = x * self.stride
                        filter_end_x = filter_start_x + self.filter_size

                        filter_start_y = y * self.stride
                        filter_end_y = filter_start_y + self.filter_size

                        # grab section of input determined by current filter location
                        # one patch per img in batch, take patch of img across all in channels/filters
                        # current in channels are determined by previous conv layers out channels (or 1 if first layer for GS imgs)
                        input_patch = inputs[img_idx, :, filter_start_y:filter_end_y, filter_start_x:filter_end_x]

                        # element-wise multiplication of 3D patch and 3D weights of corresponding filter, summing results
                        cur_filter_weights = self.weights[filter, :, :, :]
                        cur_bias = self.biases[filter]
                        conv_value = np.sum(input_patch * cur_filter_weights)
                        conv_value += cur_bias

                        out_feature_map[img_idx, filter, y, x] = conv_value
            
        return out_feature_map

    def backward(self, d_out):
        """
        Perform the backward pass through the convolutional layer.
        SEE README FOR MORE DETAILS AND MATH BEHIND THE CODE

        Args:
        - d_out: Gradient of the loss with respect to the output of this layer.
                Shape: (batch_size, out_channels, out_size, out_size)
        - inputs: The original input data to this layer during the forward pass.
                Shape: (batch_size, in_channels, in_size, in_size)
                
        Returns:
        - dX: Gradient of the loss with respect to the input data.
            Shape: (batch_size, in_channels, in_size, in_size)
        - dW: Gradient of the loss with respect to the filter weights.
            Shape: (out_channels, in_channels, filter_size, filter_size)
        - db: Gradient of the loss with respect to the biases.
            Shape: (out_channels,)
        """
        # zero pad image if self.pad is not 0
        # we pad in back prop since we want to find gradients wrt to the padded inputs
        # ensuring that borders of original image receive correct gradient contributions
        '''if self.pad > 0:
            inputs = self.zero_pad(inputs)'''
        
        dW = np.zeros_like(self.weights, dtype=np.float32) # init array for gradient of weights
        db = np.zeros_like(self.biases, dtype=np.float32) # init array for gradient of biases
        dX = np.zeros_like(self.inputs, dtype=np.float32)

        batch_size, in_channels, in_size = self.inputs.shape[:-1]
        out_channels, out_size = d_out.shape[1:-1]

        # convolve similarly to forward pass
        for img_idx in range(batch_size):               # iterate over each img in batch
            for filter_idx in range(self.out_channels): # iterate over each filter for each img
                
                # Since bias is added directly to output,
                # its grad is the sum of upstream grads (d_out) over all spatial position and batch samples for each filter
                db[filter_idx] += np.sum(d_out[img_idx, filter_idx, :, :])
                
                for y in range(out_size):               # iterate over vertically
                    for x in range(out_size):           # iterate over horizontally
                        # get current current position's gradient of loss (how much this position effects the loss)
                        cur_d_out_value = d_out[img_idx, filter_idx, y, x]

                        # get start and end indices of where the filter currently is in the img
                        filter_start_x = x * self.stride
                        filter_end_x = filter_start_x + self.filter_size

                        filter_start_y = y * self.stride
                        filter_end_y = filter_start_y + self.filter_size

                        # accumulate gradient wrt weights
                        # 
                        input_patch = self.inputs[img_idx, :, filter_start_y:filter_end_y, filter_start_x:filter_end_x]
                        dW[filter_idx] += cur_d_out_value * input_patch

                        dX[img_idx, :, filter_start_y:filter_end_y, filter_start_x:filter_end_x] += cur_d_out_value * self.weights[filter_idx]

        self.grad_W = dW
        self.grad_b = db

        # remove padding from gradient with respect to inputs
        # padding itself does not corrrespond to any real input data so padded gradients discarded
        if self.pad > 0:
            dX = dX[:, :, self.pad:-self.pad, self.pad:-self.pad]

        return dX
