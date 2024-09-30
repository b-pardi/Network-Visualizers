import numpy as np
import matplotlib.pyplot as plt

from utils.data import download_mnist, load_mnist
from utils.normalize import max_norm
from utils.activations import ReLULayer, SoftmaxLayer
from utils.interfaces import Layer

    
class Conv2D(Layer):
    def __init__(self, in_channels=1, out_channels=8, filter_size=3, stride=1, pad=0):
        self.in_channels = in_channels # channels of input data, init 3 for color (rgb) or 1 for grayscale
        self.out_channels = out_channels # represents number of kernels/filters in the conv layer
        self.filter_size = filter_size # dimensions of filter to pass over images (square)
        self.stride = stride # how many pixels we step when scanning img with kernel
        self.pad = pad # how many empty pixels we 'pad' the border with
        
        # normal xavier/glorot initialization to help maintain stability of activation variance. Likely overkill for mnist digits, but it's for pedagogical purposes
        # weights ~ N (mu, sigma); mu is 0, sigma is sqrt(2 / n); n is number of input and output neurons to current layer
        n_in = in_channels * filter_size **2
        n_out = out_channels
        variance = np.sqrt(2 / (n_in + n_out))

        # size ordered as such allows for easier mat mul (matches input dimensions)
        self.weights = np.random.normal(0, variance, size=(out_channels, in_channels, filter_size, filter_size))
        self.biases = np.zeros((out_channels,), dtype=np.float32) # one bias term for each filter (out channel)

    def zero_pad(self, inputs):
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
        # receives 4D array (batch_size, in_channels, in_size, in_size) -> [64, 1, 28, 28]
        
        # zero pad image if self.pad is not 0
        if self.pad > 0:
            inputs = self.zero_pad(inputs)

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
                        cur_bias = self.bias[filter]
                        conv_value = np.sum(input_patch * cur_filter_weights)
                        conv_value += cur_bias

                        out_feature_map[img_idx, filter, y, x] = conv_value
            
        return out_feature_map

    def backward(self, d_out, inputs):
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
        if self.pad > 0:
            inputs = self.zero_pad(inputs)
        
        dW = np.zeros_like(self.weights, dtype=np.float32) # init array for gradient of weights
        db = np.zeros_like(self.biases, dtype=np.float32) # init array for gradient of biases
        dX = np.zeros_like(inputs, dtype=np.float32)

        batch_size, in_channels, in_size = inputs.shape[:-1]
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
                        input_patch = inputs[img_idx, :, filter_start_y:filter_end_y, filter_start_x:filter_end_x]
                        dW[filter_idx] += cur_d_out_value * input_patch

                        dX[img_idx, :, filter_start_y:filter_end_y, filter_start_x:filter_end_x] += cur_d_out_value * self.weights[filter_idx]

        # remove padding from gradient with respect to inputs
        # padding itself does not corrrespond to any real input data so padded gradients discarded
        if self.pad > 0:
            dX = dX[:, :, self.pad:-self.pad, self.pad:-self.pad]

        return dX, dW, db

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
                        max_grad = mask * d_out[img_idx, filter_idx, y // self.stride, x // self.stride]
                        dX[img_idx, filter_idx, y:y + self.filter_size, x:x + self.filter_size] += max_grad
        return dX
        

# Flatten Layer
class FlattenLayer(Layer):
    def forward(self, inputs):
        """
        Pseudocode:
        - Reshape the multi-dimensional input into a 2D array.
        """
        pass

    def backward(self, grad_output):
        """
        Pseudocode:
        - Reshape gradient output back to the input's original shape.
        """
        pass

# Fully Connected Layer (Dense)
class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        """
        Pseudocode:
        - Initialize weights and biases.
        - `input_size`: Number of input features.
        - `output_size`: Number of output neurons.
        """
        pass

    def forward(self, inputs):
        """
        Pseudocode:
        - Apply affine transformation: z = xW + b
        - Store inputs for backward pass.
        """
        pass

    def backward(self, grad_output):
        """
        Pseudocode:
        - Compute gradient of loss w.r.t. weights, biases, and inputs.
        - Update weights and biases using learning rate.
        """
        pass

# Neural Network Class
class CNN:
    def __init__(self, layers):
        """
        Pseudocode:
        - Initialize network with a list of layers.
        """
        pass
    
    def forward(self, X):
        """
        Pseudocode:
        - Pass input through each layer sequentially.
        - Return final output.
        """
        pass

    def backward(self, loss_grad):
        """
        Pseudocode:
        - Pass gradient backward through each layer.
        """
        pass

    def train(self, X_train, y_train, epochs, learning_rate):
        """
        Pseudocode:
        - Iterate over epochs.
        - For each batch in training data:
            - Forward pass.
            - Compute loss and loss gradient.
            - Backward pass.
            - Update parameters with gradients.
        """
        pass

    def predict(self, X):
        """
        Pseudocode:
        - Perform forward pass through the network.
        - Return predictions.
        """
        pass

# Loss Function
def compute_loss(y_pred, y_true):
    """
    Pseudocode:
    - Compute loss (e.g., Cross-Entropy Loss).
    """
    pass

def compute_loss_gradient(y_pred, y_true):
    """
    Pseudocode:
    - Compute gradient of loss w.r.t. network output.
    """
    pass

def plot_mnist_digit(train_img, train_label, test_img, test_label):
    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(train_img, cmap='gray')
    plt.title(f"TRAIN DIGIT: {train_label}")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(test_img, cmap='gray')
    plt.title(f"TEST DIGIT: {test_label}")
    plt.axis('off')

    plt.show()

def run_mnist_training():
    mnist_md5_dict = {
        'train_images': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
        'train_labels': 'd53e105ee54ea40749a09fcbcd1e9432',
        'test_images': '9fb629c4189551a2d022fa330f9573f3',
        'test_labels': 'ec29112dd5afa0611ce80d1b7f02629c'
    }

    data_dir_dict = download_mnist("res/mnist_data/")
    x_train, y_train, x_test, y_test = load_mnist(data_dir_dict, mnist_md5_dict)

    # verify images loaded correct plotting a random one
    random_train_idx = np.random.randint(0, len(x_train))
    random_test_idx = np.random.randint(0, len(x_test))
    plot_mnist_digit(x_train[random_train_idx], y_train[random_train_idx], x_test[random_test_idx], y_test[random_test_idx])

    # test code
