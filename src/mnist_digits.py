import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl

from utils.data import download_mnist, load_mnist, onehot_encode
from utils.normalize import constant_norm
from utils.activations import ReLULayer, SoftmaxLayer
from utils.interfaces import Layer
from utils.initializers import xavier_glorot_normal_init
from utils.errs import cce_loss, cce_loss_gradient

    
class Conv2D(Layer):
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

# Fully Connected Layer (Dense)
class DenseLayer(Layer):
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

# Neural Network Class
class CNN:
    """
    A simple self implemented CNN model for multi-class classification (e.g., MNIST).
    
    This CNN consists of Conv2D, ReLU, MaxPooling, Flatten, and Dense layers,
    with a final softmax activation layer to classify

    for comments below describing size, b refers to batch size, and initial image size is based on 28x28 mnist images
    """
    def __init__(self, num_classes):
        # network initialized as list of layers
        self.layers = [
            # first layer 2D convolution, 1 in channel for grayscale, 8 out channels for 8 filters, 3x3 filter
            # shape in: b, 1, 28, 28
            Conv2D(in_channels=1, out_channels=8, filter_size=3, stride=1),
            # shape out: b, 8, 26, 26; with stride of 1, dims reduced by filter_size - 1, and 8 filters applied

            # activate filter weights
            ReLULayer(), # does not effect dimensions

            # reduce image dimensionality keeping only important parts with max pooling
            # shape in: b, 8, 26, 26 from prev conv2d layer
            MaxPoolingLayer(filter_size=2, stride=2),
            # shape out: b, 8, 13, 13 dims reduce by factor of (2-1) // 2 + 1 = 2

            # second layer 2D convolution, 8 in channels from previous conv out, 16 out channels for 16 new filters
            # shape in: b, 8, 13, 13 from previous pooling layer
            Conv2D(in_channels=8, out_channels=16, filter_size=3, stride=1),
            # shape out: b, 16, 11, 11 dims reduce by 2, 16 filters used in this conv layer

            # activate filter weights
            ReLULayer(), # dims unaffected by relu

            # pooling again
            # shape in: b, 8, 11, 11 from previous conv layer
            MaxPoolingLayer(filter_size=2, stride=2),
            # shape out: b, 8, 5, 5 reduced by factor of 2 again (integer division)

            # flatten to prepare for dense layer
            FlattenLayer(),

            # to find fc in_features, look at most recent conv2d's out channels,
            # and dimension of feature maps that conv and pool layers have reduced the original dimensions from
            DenseLayer(in_features=16 * 5 * 5, out_features=256), # funnel down to fewer neurons before going straight to num_classes

            ReLULayer(),
            
            DenseLayer(in_features=256, out_features=num_classes),

            SoftmaxLayer()
        ]
    
    def forward(self, x):
        """
        Forward Pass: propogate through layers and call each layer's forward function

        Args:
        - x (ndarray): The input data (e.g., batch of images).
        
        Returns:
        - x (ndarray): The final output after passing through all layers.
        """
        #print("FORWARD")
        for layer in self.layers:
            #layer_name = layer.__class__.__name__
            #print(f"{layer_name} input shape: {x.shape}")
            x = layer.forward(x)
            #print(f"{layer_name} output shape: {x.shape}")

        return x

    def backward(self, d_out):
        """
        Backward pass: Propagate the gradient through each layer in reverse order.
        
        Args:
        - d_out (ndarray): The gradient of the loss with respect to the output.
        
        Returns:
        - Gradients are computed and stored in the layers for parameter updates.
        """
        #print("BACKWARD")

        for layer in reversed(self.layers):
            #layer_name = layer.__class__.__name__
            #print(layer_name, f"before: grad out type: {type(d_out)}")
            d_out = layer.backward(d_out)
            #print(layer_name, f"after: grad out type: {type(d_out)}")

    def train(self, x_train, y_train, epochs, lr, batch_size):
        """"
        Simple training loop using gradient descent.
        
        Args:
        - X_train (ndarray): Training data (e.g., MNIST images).
        - y_train (ndarray): One-hot encoded true labels for the training data.
        - num_epochs (int): Number of training epochs.
        - learning_rate (float): Learning rate for gradient descent.
        """
        num_train_samples = len(x_train)
        num_batches = num_train_samples // batch_size

        for epoch in range(epochs):
            # shuffle data each epoch so batches equally represent dataset
            permutation = np.random.permutation(num_train_samples)
            x_train = x_train[permutation]
            y_train = y_train[permutation]

            epoch_loss = 0.
            with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True) as progress_bar:
                for i in range(0, num_train_samples, batch_size):
                    batch_end_idx = min(i+batch_size, num_train_samples)
                    
                    # this index method ensures sample remains 2D instead of being flattened
                    x_batch = x_train[i:batch_end_idx]  # batch of images
                    y_batch = y_train[i:batch_end_idx]  # batch of labels (one-hot)

                    probs = self.forward(x_batch) # forward pass

                    loss = cce_loss(probs, y_batch) # find loss of prediction
                    epoch_loss += loss

                    grad_loss = cce_loss_gradient(probs, y_batch) # gradient of loss

                    # start the backward loop with the loss of the prediction (dL/dx)
                    self.backward(grad_loss)

                    # gradient descent
                    # update all weights and biases based each layer's stored grads
                    for layer in self.layers:
                        if hasattr(layer, 'weights'):
                            layer.weights -= lr * layer.grad_W
                            layer.biases -= lr * layer.grad_b

                    # Update the progress bar and show the average loss per sample
                    avg_loss = epoch_loss / (batch_end_idx)  # Calculate the average loss so far
                    progress_bar.set_postfix({'loss': avg_loss})  # Display the loss in the progress bar
                    progress_bar.update(1)  # Increment the progress bar by 1

    def predict(self, x):
        """
        Run a forward pass through the network to make a prediction
        """

        logits = self.forward(x)
        return np.argmax(logits, axis=1)
    
    def save_model(self, fn):
        with open(fn, 'wb') as out_file:
            pkl.dump(self, out_file)


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

def run_mnist():
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
    #plot_mnist_digit(x_train[random_train_idx], y_train[random_train_idx], x_test[random_test_idx], y_test[random_test_idx])

    # normalize pixels values to be between 0 and 1
    x_train = constant_norm(x_train, c=255.0)
    x_test = constant_norm(x_test, c=255.0)

    # reshape to have batch sizes
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)

    # one hot encode labels
    y_train = onehot_encode(y_train, num_classes=10)

    # create CNN object and run training
    cnn = CNN(num_classes=10)
    cnn.train(
        x_train,
        y_train,
        epochs=10,
        lr=0.01,
        batch_size=32
    )
    
    cnn.save_model("models/mnist_cnn.pkl")