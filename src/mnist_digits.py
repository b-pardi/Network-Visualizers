import numpy as np
import matplotlib.pyplot as plt

from utils.data import download_mnist, load_mnist
from utils.normalize import max_norm

# Enforce an interface to ensure all derived classes have forward and backward methods
class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError
    
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
        self.biases = np.zeros((out_channels,)) # one bias term for each filter (out channel)

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
        out_feature_map = np.zeros((batch_size, self.out_channels, out_size, out_size))

        # convolution process
        for img_idx in range(batch_size):           # iterate over all imgs in batch
            for filter in range(self.out_channels): # iterate over each filter (output channel)
                for y in range(out_size):         # iterate over vertically
                    for x in range(out_size):     # iterate over horizontally
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

    def backward(self, grad_output):
        """
        Pseudocode:
        - Inputs: numpy array of shape (batch_size, in_channels, height, width).
        - If padding is not zero, apply zero-padding to the input.
        - Create an output array of the correct size based on the input size,
          stride, and padding.
        - For each filter:
            - Slide the filter over the input, applying element-wise multiplication 
              and summing the results to form the output feature map.
        - Store the input and output for use in the backward pass.
        - Return the output.
        """
        pass

# ReLU Activation Layer
class ReLULayer(Layer):
    def forward(self, inputs):
        """
        Pseudocode:
        - Apply ReLU function: f(x) = max(0, x)
        - Store mask of positive values for backward pass.
        """
        pass

    def backward(self, grad_output):
        """
        Pseudocode:
        - Backpropagate only through positive inputs using the mask.
        """
        pass

# Max Pooling Layer
class MaxPoolingLayer(Layer):
    def __init__(self, pool_size, stride):
        """
        Pseudocode:
        - Set pooling window size and stride.
        """
        pass
    
    def forward(self, inputs):
        """
        Pseudocode:
        - Slide pooling window over input.
        - Take the maximum value in each window.
        - Store indices of max values for backward pass.
        """
        pass

    def backward(self, grad_output):
        """
        Pseudocode:
        - Distribute gradients to the locations of max values.
        """
        pass

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
    