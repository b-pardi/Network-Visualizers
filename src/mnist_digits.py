import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import json
import os

from layers.static.activations import ReLULayer, SoftmaxLayer
from layers.static.pooling import MaxPoolingLayer
from layers.static.flatten import FlattenLayer
from layers.learnable.convolutions import Conv2D
from layers.learnable.dense import DenseLayer

from utils.data import download_mnist, load_mnist, onehot_encode, stratified_split, select_fraction_of_data, downsample_square_images
from utils.normalize import constant_norm
from utils.errs import cce_loss, cce_loss_gradient
from utils.optimizers import AdamOptimizer
from utils.metrics import accuracy_score, precision_score, f1_score

# Neural Network Class
class CNN:
    """
    A simple self implemented CNN model for multi-class classification (e.g., MNIST).
    
    This CNN consists of Conv2D, ReLU, MaxPooling, Flatten, and Dense layers,
    with a final softmax activation layer to classify

    network is assembled based on the "config/cnn_params.json" file
    """
    def __init__(self, params):
        self.params = params
        num_classes = params["num_classes"]
        input_size = params["input_size"]
        in_channels = params["in_channels"]
        layer_assembly = params["layers"]

        # network initialized as list of layers
        self.layers = []

        # flag used to indicate to dense layers that the data was just flattened
        # if it was just flattened additional calculations must be made to determine in_features
        # else there will be out_features from the previous dense layer ready to use as in_feature for this one
        just_flattened = False
        running_in_features = None
        
        running_in_size = input_size
        running_in_channels = in_channels

        # loop through layer assembly and and assemble network according to it
        for layer in layer_assembly:
            if layer['type'] == 'conv':
                # Add Conv2D layer
                self.layers.append(Conv2D(
                    in_channels=running_in_channels,
                    out_channels=layer['out_channels'],
                    filter_size=layer['filter_size'],
                    stride=layer['stride'],
                    pad=layer['padding']
                ))
                
                # Calculate the new output size after the convolution
                running_in_size = self.calculate_output_size(running_in_size, layer['filter_size'], layer['stride'], layer['padding'])
                running_in_channels = layer['out_channels']  # Update the in_channels for the next layer
                
            if layer['type'] == 'maxpool':
                # Add MaxPoolingLayer
                self.layers.append(MaxPoolingLayer(
                    filter_size=layer['filter_size'],
                    stride=layer['stride'],
                ))

                # Calculate the new output size after pooling
                running_in_size = self.calculate_output_size(running_in_size, layer['filter_size'], layer['stride'])
                
            if layer['type'] == 'relu':
                self.layers.append(ReLULayer()) # dims unaffected by relu

            if layer['type'] == 'flatten':
                self.layers.append(FlattenLayer())
                just_flattened = True # let dense layer know that it is receiving a flattened layer and to make appropriate calculations

            if layer['type'] == 'dense':
                if just_flattened:
                    # the first dense layer takes in the flatten layer
                    # meaning there is a feature map of (running_in_size x running_in_size) for each running_channel
                    # that corresponding number of in_features is calculated here
                    in_features = running_in_channels * running_in_size * running_in_size
                    just_flattened = False
                else:
                    # if not just flattened, the previous dense layer gave us out features to use
                    in_features = running_in_features

                # Add DenseLayer
                self.layers.append(DenseLayer(
                    in_features=in_features,
                    out_features=layer['out_features']
                ))

                # Update running_in_features for the next dense layer
                running_in_features = layer['out_features']

            if layer['type'] == 'softmax':
                self.layers.append(SoftmaxLayer())

    @staticmethod   
    def calculate_output_size(input_size, filter_size, stride, padding=0):
        """
        Calculate the output dimension after a convolution or pooling layer.
        
        Parameters:
            input_size (int): The height/width of the input (assuming square images).
            filter_size (int): The size of the filter (kernel size).
            stride (int): The stride of the filter.
            padding (int): The number of pixels to pad on each side of the input (default is 0).
        
        Returns:
            int: The output size (height/width) after applying the layer.
        """
        return (input_size - filter_size + 2 * padding) // stride + 1
        
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

    def train(
            self,
            x_train,
            y_train,
            x_val,
            y_val,
            validation_subset_ratio,
            epochs,
            patience,
            batch_size,
            optimizer
        ):
        """"
        Training loop using Adam optimizer for minibatch SGD with a train and validation loop.
        
        Args:
        - x_train, y_train: Training data and labels
        - x_val, y_val: Validation data and labels for early stopping
        - validation_subset_size: Number of validation samples to use each epoch
        - epochs: Number of training epochs
        - patience: Number of epochs to wait for improvement before stopping early
        - batch_size: Batch size for mini-batch gradient descent
        - optimizer: AdamOptimizer instance
        """
        num_train_samples = len(x_train)
        num_batches = num_train_samples // batch_size

        # init adam optimizer for learnable layers (layers with weights and biases)
        optimizer.init_params(self.layers)

        best_val_loss = np.inf # track best loss to determine early stop
        patience_counter = 0 # track n_epochs without much improvement for early stop

        for epoch in range(epochs):
            # shuffle data each epoch so batches equally represent dataset
            permutation = np.random.permutation(num_train_samples)
            x_train = x_train[permutation]
            y_train = y_train[permutation]

            epoch_loss = 0.
            total_train_acc, total_train_prec, total_train_f1 = 0, 0, 0  # Initialize cumulative metrics

            with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{epochs} [TRAINING]", dynamic_ncols=True) as train_bar:
                for i in range(0, num_train_samples, batch_size):
                    # find index of batch end while preventing i from reaching outside of all train samples
                    batch_end_idx = min(i+batch_size, num_train_samples)
                    
                    x_batch = x_train[i:batch_end_idx]  # batch of images
                    y_batch = y_train[i:batch_end_idx]  # batch of labels (one-hot)

                    probs = self.forward(x_batch) # forward pass

                    loss = cce_loss(probs, y_batch) # find loss of prediction
                    epoch_loss += loss

                    grad_loss = cce_loss_gradient(probs, y_batch) # gradient of loss

                    # start the backward loop with the loss of the prediction (dL/dx)
                    self.backward(grad_loss)

                    # replace traditional SGD with adam optimized SGD
                    # still updates the weights and biases, just more effectively
                    optimizer.update()

                    # training metrics
                    y_pred = np.argmax(probs, axis=1)
                    y_true = np.argmax(y_batch, axis=1)
                    total_train_acc += accuracy_score(y_true, y_pred)
                    total_train_prec += precision_score(y_true, y_pred)
                    total_train_f1 += f1_score(y_true, y_pred)

                    # Update the progress bar and show the average loss per sample
                    batches_processed = i // batch_size + 1
                    train_bar.set_postfix({
                        'loss': epoch_loss / batches_processed,
                        'accuracy': total_train_acc / batches_processed,
                        'precision': total_train_prec / batches_processed,
                        'f1': total_train_f1 / batches_processed
                    })  # Display metrics in the progress bar
                    train_bar.update(1)  # Increment the progress bar by 1

            # validation phase using random smaller subset of validation dataset
            val_acc, val_prec, val_f1 = 0, 0, 0
            val_loss = 0.
            validation_subset_size = int(validation_subset_ratio * len(x_val))
            val_indices = np.random.choice(len(x_val), min(validation_subset_size, len(x_val)), replace=False)
            x_val_subset = x_val[val_indices]
            y_val_subset = y_val[val_indices]

            # validation progress bar
            with tqdm(total=validation_subset_size // batch_size, desc=f"Epoch {epoch+1}/{epochs} [VALIDATION]", dynamic_ncols=True) as val_bar:
                for i in range(0, validation_subset_size, batch_size):
                    # batch dataset
                    batch_end_idx = min(i+batch_size, validation_subset_size)
                    x_batch_val = x_val_subset[i:batch_end_idx]
                    y_batch_val = y_val_subset[i:batch_end_idx]

                    # forward pass on validation data
                    val_probs = self.forward(x_batch_val)
                    val_loss += cce_loss(val_probs, y_batch_val)

                    # validation metrics
                    y_pred_val = np.argmax(val_probs, axis=1)
                    y_true_val = np.argmax(y_batch_val, axis=1)
                    val_acc += accuracy_score(y_true_val, y_pred_val)
                    val_prec += precision_score(y_true_val, y_pred_val)
                    val_f1 += f1_score(y_true_val, y_pred_val)

                    # validation progress bar with loss and metrics
                    val_batches_processed = (i // batch_size + 1)
                    val_bar.set_postfix({
                        'loss': val_loss / val_batches_processed,
                        'accuracy': val_acc / val_batches_processed,
                        'precision': val_prec / val_batches_processed,
                        'f1': val_f1 / val_batches_processed
                    })
                    val_bar.update(1)

            # get most recent stats so when model finishes training, these will be saved in the pkl object
            self.final_train_acc = total_train_acc / batches_processed
            self.final_train_prec = total_train_prec / batches_processed
            self.final_train_f1 = total_train_f1 / batches_processed
        
            self.final_val_acc = val_acc / val_batches_processed
            self.final_val_prec = val_prec / val_batches_processed
            self.final_val_f1 = val_f1 / val_batches_processed

            # early stopping check
            avg_val_loss = val_loss / validation_subset_size
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early Stopping Triggered at Epoch: {epoch+1} after {patience_counter} epochs with no improvement")
                    print(f"Validation Loss at stopping: {avg_val_loss}, Best Validation Loss: {best_val_loss}")
                    break

    def predict(self, x):
        """
        Run a forward pass through the network to make a prediction
        """

        probs = self.forward(x)
        return np.argmax(probs, axis=1)
    
    def predict_prob(self, x):
        """
        Run a forward pass through the network to make a prediction
        return the probability of its predicition instead
        """

        return self.forward(x)
    
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

def gather_mnist_data(cnn_params):
    mnist_md5_dict = {
        'train_images': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
        'train_labels': 'd53e105ee54ea40749a09fcbcd1e9432',
        'test_images': '9fb629c4189551a2d022fa330f9573f3',
        'test_labels': 'ec29112dd5afa0611ce80d1b7f02629c'
    }

    data_dir_dict = download_mnist("res/mnist_data/")
    x_train, y_train, x_test, y_test = load_mnist(data_dir_dict, mnist_md5_dict)

    print(f"Initial Dataset Shapes: x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}")

    # grab a smaller portion of dataset, all 60k images are overkill for this project's purpose
    x_train, y_train = select_fraction_of_data(x_train, y_train, 10, fraction=cnn_params['percent_of_train_data_to_use'])
    x_test, y_test = select_fraction_of_data(x_test, y_test, 10, fraction=cnn_params['percent_of_train_data_to_use'])

    # downsample images for same reason as above
    x_train = downsample_square_images(x_train, cnn_params['input_size'])
    x_test = downsample_square_images(x_test, cnn_params['input_size'])

    # verify images loaded correct plotting a random one
    random_train_idx = np.random.randint(0, len(x_train))
    random_test_idx = np.random.randint(0, len(x_test))
    #plot_mnist_digit(x_train[random_train_idx], y_train[random_train_idx], x_test[random_test_idx], y_test[random_test_idx])

    # normalize pixels values to be between 0 and 1
    x_train = constant_norm(x_train, c=255.0)
    x_test = constant_norm(x_test, c=255.0)

    # reshape to have batch sizes
    image_size = x_train.shape[-1]
    x_train = x_train.reshape(-1, 1, image_size, image_size)
    x_test = x_test.reshape(-1, 1, image_size, image_size)
    print(x_train.shape, y_train.shape)

    # one hot encode labels
    y_train = onehot_encode(y_train, num_classes=10)
    y_test = onehot_encode(y_test, num_classes=10)

    # split train data into 10% test, ensuring equal samples of all classes represented, but still shuffled
    x_train, y_train, x_val, y_val = stratified_split(x_train, y_train, val_size=cnn_params['percent_train_data_to_use_for_validation'], num_classes=10, random_state=123)

    print(f"Post-Processed Dataset Shapes: x_train: {x_train.shape}, y_train: {y_train.shape}, x_val: {x_val.shape}, y_val: {y_val.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}")

    return x_train, y_train, x_val, y_val, x_test, y_test

def run_mnist_train(config_fp):
    if not os.path.isfile(config_fp):
        raise FileNotFoundError(f"ERROR: found no model object file at {config_fp}")

    # load the parameters for the CNN
    with open(config_fp, 'r') as config_file:
        cnn_params = json.load(config_file)

    # download/grab and preprocess images
    x_train, y_train, x_val, y_val, _, _ = gather_mnist_data(cnn_params)

    # optimizer object
    optim = AdamOptimizer(
        lr=0.01,        # initial learning rate
        beta1=0.9,      # decay rate for first moment; 0.9 -> care more about more recent gradients
        beta2=0.999,    # decay rate for second moment; 0.999 -> keep longer history of past gradients
        epsilon=1e-8    # prevent divide by 0
    )

    # create CNN object using config file and run training
    cnn = CNN(cnn_params)
    cnn.train(
        x_train,
        y_train,
        x_val,
        y_val,
        validation_subset_ratio=cnn_params['validation_subset_ratio'], # how much of the validation set to use each epoch (randomly selects this fraction from val set)
        epochs=cnn_params['epochs'], # how many training/validation iterations
        patience=cnn_params['patience'], # how many epochs without seeing improvement before stopping
        batch_size=cnn_params['batch_size'], # how many images to process at once -> SGD occurs once for each batch
        optimizer=optim
    )
    
    cnn.save_model(f"models/{cnn_params['model_output_filename']}.pkl")

def run_mnist_inference(model_fp):
    if not os.path.isfile(model_fp):
        raise FileNotFoundError(f"ERROR: found no model object file at {model_fp}")
    
    with open(model_fp, 'rb') as obj_file:
        cnn = pkl.load(obj_file)

    cnn_params = cnn.params

    _, _, _, _, x_test, y_test = gather_mnist_data(cnn_params)

    indices = np.random.choice(len(x_test), 4, replace=False)  # Randomly select 4 indices
    x_random_4 = x_test[indices]  # Select the images based on the random indices
    y_true_random_4 = y_test[indices]  # Select the corresponding true labels (one-hot encoded)

    y_pred_random_4 = cnn.predict(x_random_4)

    # convert the one-hot encoded true labels to class indices
    y_true_random_4_indices = np.argmax(y_true_random_4, axis=1)

    # plot the 4 images with titles as predicted and true labels
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))  # Create 4 subplots

    for i, ax in enumerate(axs):
        ax.imshow(x_random_4[i].squeeze(), cmap='gray')
        ax.set_title(f"Pred: {y_pred_random_4[i]}, True: {y_true_random_4_indices[i]}")
        ax.axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()