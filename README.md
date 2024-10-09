# Network-Visualizers
Personal implementations of various networks without any ML libraries and visualizations for each. Visuals all show *live* as the network trains or infers on data, giving users a deep insight as to what goes on under the hood of these algorithms as they run on real data.
Currently only the simple Neural Network to solve the XOR problem is implemented with visuals, and the CNN trained on MNIST is implemented with visuals in progress. Further implementations are also planned (see Planned Implementations section)

# Table of Contents

- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Handling CLI Arguments](#handling-cli-arguments)
  - [Example Usage](#example-usage)
- [Usage Instructions](#usage-instructions)
  - [XOR (Exclusive OR) Problem Neural Net](#xor-exclusive-or-problem-neural-net-usage)
  - [Convolutional Neural Network - MNIST Handwritten Digits](#convolutional-neural-network---mnist-handwritten-digits-usage)
- [Currently Implemented Network Deep Dives](#currently-implemented-network-deep-dives)
  - [XOR (Exclusive OR) Problem Neural Net](docs/xor.md)
  - [Convolutional Neural Network - MNIST Handwritten Digits](docs/mnist_cnn.md)
- [Planned Implementations](#planned-implementations)
- [Contributing](#contributing)

---

# Project Overview

I wanted to make this project to open the 'black box' we call Machine Learning. Even after solving several problems with various machine learning methods it still seemed like a mystery to me, it still felt like just magic. So I sought to implement various basic networks and algorithms using only NumPy to figure this out. After that I wasn't fully satisfied, and wanted to *see* what was happening, and really connect the conceptual aspects and math behind machine learning to the code. I've seen countless visuals, plots, diagrams, etc. that are supposed to help understand, and they did help, but something was missing. Most were still abstracted since they were created to *describe* what was happening, but I wanted to see what is *actually* happening as it happens. This project is and will continue to be a culmination of efforts to see what actually happens inside of various machine learning algorithms *as it happens*.

# Setup and Installation

1. Clone the repository:
    - `git clone https://github.com/yourusername/Network-Visualizers.git`
    - `cd Network-Visualizers`

2. Install the required dependencies (virtual environment recommended):
    - `python -m venv venv`
    - Windows: `.\venv\Scripts\activate` Mac/Linux: `source venv/bin/activate`
    - `pip install -r requirements.txt`

3. Run the desired neural network visualizer:

    - **XOR Problem**:
    `python main.py --xor`

    - **MNIST Digit Recognition (WIP)**:
    `python main.py --mnist`

4. To exit training early, press 'q'

# Handling CLI Arguments

The script accepts several command-line arguments to control the visualization and configuration of the networks:

- `--xor`: Run the XOR classifier
    - `-hn`, `--hidden_neurons`: Number of neurons in the hidden layer (default is 3)
    - `-ner`, `--num_epochs_refresh_visualizer`: Number of epochs per refreshing of visualizers (affects the speed of training; default is 100 for xor)
    - `e`, `--epochs`: Number of epochs to train selected model
    
- `-md`, `--mnist_digits`: Run the MNIST digit classifier (planned implementation)
    - `-m`, `--mode`:
        - `train`: will run the fully self implemented Convolutional Neural Network training loop on the MNIST dataset and show corresponding visuals (WIP). Uses default config file `config/default_cnn_params.json`
            - `-uc`, `--user_config`: Opt to use the `config/user_cnn_params.json` file to assemble the network when training. This is the customizeable CNN configuration the user can alter to change hyperparameters and the assembly of network layers without having to define a custom CNN class
        - `infer`: Infer on a pretrained CNN created from this network and show corresponding visuals. Pretrained CNN using default config file
            - `-um`, `--user_model` <file_path>: Opt to use a model the user has previously trained with this project's CNN for inference and show the visualizations that go with it. Will read from file path spec'd in the argument. Note, if the user has not trained there own yet, or the name/location of this pickle object file was changed, it will throw a file not found error. Additionaly please note this is not a weight or tensors file, this a file of the whole CNN object, so that the parameters used for training can be read on inference. This allows things like input image size to be read so that the images can be scaled to the same size that the model was trained on.


**Note:** args shown above that are indented in the list indicate that they are only relevant to the parent arg. i.e. `-hn` is only used for `--xor`

## Example Usage

To run the XOR neural network with a custom number of hidden neurons, epochs, and visualizer refresh rate:

`python main.py --xor -hn 3 -ner 200 -e 40000`

To run the MNIST classifier and train a CNN using the default configuration:

`python main.py --mnist_digits -m train`

To run the MNIST classifier and train a CNN using a user-defined configuration:

`python main.py --mnist_digits -m train -uc`

To infer on the MNIST dataset using the default pretrained CNN model:

`python main.py --mnist_digits -m infer`

To infer on the MNIST dataset using a custom user-trained model:

`python main.py --mnist_digits -m infer -um models/custom_mnist_model.pkl`

# Currently Implemented Network Deep Dives

Deep dives describe in great detail the problem the network is trying to solve and how it solves it, as well as the math behind how the network solves it and the formulas that led to the code created.
All deep dives are in separate .md files to keep things organized, they can be found in the `docs/` folder, or here below:

- [XOR (Exclisive Or) Problem Neural Network](docs/xor.md)
- [Convolutional Neural Network - MNIST Handwritten Digits](docs/mnist_cnn.md)
    - Note that the MNIST CNN is a work in progress

# Usage Instructions

## XOR (Exclusive OR) Problem Neural Net Usage

This neural network consists of 3 layers (input, hidden, and output) to solve the XOR gate problem. The XOR problem is a fundamental concept in machine learning, demonstrating non-linearly separable classification.

### Running the Network

To run the XOR network, use the command-line arguments described above. You can adjust the number of hidden neurons to see different visualizations with the `--hidden_neurons` argument, andor modify the `--num_epochs_refresh_visualizer` argument to control the visualization update speed (increase it if your computer struggles to handle frequent updates).

The number of hidden neurons affects the available plots due to dimensional constraints. For example, the feature space plot is only available for 2 or 3 hidden neurons, as visualizing higher-dimensional feature spaces is not feasible. Below is a list of visualizations shown for different numbers of hidden neurons:

### Description of Visuals

- **More than 3 hidden neurons**:
    - **Network Architecture**: Shows the structure of the neural network, including the input, hidden, and output layers, and the connections between them. Also shows each node's input, activation, and bias values as they change overtime
    - **Loss**: Displays the Mean Squared Error (MSE) loss value over time during training, showing how well the network is learning. This will decrease in time as the network converges on a solution
    - **Decision Boundaries**: Visualizes the boundaries learned by the network to separate different classes (i.e., how the network splits the input space).
    - **Gradient Norms**: Plots the magnitude of gradients during backpropagation, helping you understand the optimization process and convergence behavior. It signifies how much weights are being changed each epoch, where a steeper curve shows the weights changing a lot, and a curve approaching horizontal shows they're not changing much, indicating the network is either converging on a solution, or potentially stuck.

- **3 Hidden Neurons** (`-hn 3`):
    - The above plots, plus:
    - **3D Feature Space**: A 3D plot of the activated values of neurons in the hidden layer, visualizing how the network transforms the input space. This plot shows how the network moves the points in feature space such that they become linearly separable, and in the case of 3 neurons (3D) we will see a plane show the classes as linearly separable, with a logistic regression performed on feature points to confirm.

- **2 Hidden Neurons** (`-hn 2`):
    - All the above plots, plus:
    - **2D Feature Space**: A 2D plot of neuron activation values, showing how the network maps inputs into a new space for classification. This is conceptually the same as the 3D feature space plot above, however since we have 2 hidden neurons we have 2 points in feature space, and so the plot is 2D. In addition we have a logistic regression line instead of a plane to separate the classes
    - **3D Gradient Trajectory on the Loss Surface**: A 3D visualization of the loss surface and weight trajectory during training, showing how the network tunes the weights to seek a minimum loss on the loss surface. The red dots show the path of gradient descent each epoch and how it seeks the "lowest point" on the loss surface.

### Important Notes on XOR Problem Usage
It is most insightful to run the XOR network with 2 or 3 hidden neurons, as this allows for visualizing the feature space mapping of the input points and in the case of 2 neurons the gradient trajectory. For networks with more than 3 hidden neurons, these plots are omitted due to dimensionality visualization limitations, but the network will still train with other plots visible.

For more math and theory behind the problem and to understand how networks find solutions, see the deep dive here: [XOR (Exclusive OR) Problem Neural Net](docs/xor.md)

---

## Convolutional Neural Network - MNIST Handwritten Digits Usage

**THIS IS A WORK IN PROGRESS. NETWORK IS COMPLETE AND USER CAN TRAIN AND INFER WITH IT BUT VISUALS ARE NOT YET IMPLEMENTED**

This is a fully self implemented, customizeable Convolutional Neural Network with all the layers needed to assemble a network however the user desires.

You are able to train your own CNN on MNIST data or infer on either a pretrained model or your own trained CNN from this project. Both have visualizations described below:

### Training Visuals
- WIP

### Inference Visuals
- WIP

### Configuring Your Own Network

The CNN model is assembled based on the configuration file, defaulting to `config/default_cnn_params.json`, but can be customized with a user-defined configuration by altering `config/default_cnn_params.json`. This section outlines how the parameters in the configuration file work, their effect on the network, and how to correctly assemble a network using this setup. This will also cover how to assemble layers yourself using purely a config file without having to implement your own class to assemble a network.

### Network Parameters

The following parameters are defined in the configuration file:

- **epochs**: The number of times the entire training dataset is passed through the network. Higher values allow the model to learn better but increase training time. (Default is 10)
  
- **patience**: Used for early stopping. If the validation loss does not improve after a specified number of epochs, training is stopped early to prevent overfitting. Set this to be greater than or equal to your epochs to disable early stopping. (Default is 2)

- **batch_size**: The number of samples that the network processes at one time before updating the weights. Smaller batch sizes give more updates but can be computationally expensive, while larger batch sizes may miss finer details in the data. (Default is 32)

- **validation_subset_ratio**: The proportion of the validation set used for evaluating the model during training. A value of 0.5 means half the validation set is randomly selected. **Note this is NOT the same as validation set size, that is detailed below** (Default is 0.5)

- **percent_of_train_data_to_use**: A percentage of the training data to use during the training process. If you want to train on a subset of the data (e.g., 20%), set this parameter accordingly. MNIST training set consists of 60,000 images, this is largely overkill and takes longer. (Default is 0.2)

- **percent_train_data_to_use_for_validation**: A percentage of the training data used specifically for validation during training. This is how much of the training data to set aside for validation purposes. (Default is 0.2)

- **num_classes**: The number of classes for classification (e.g., 10 for MNIST digits). (Default is 10 don't change this if still using on MNIST digits)

- **input_size**: The size of the input images. For example, `14` means the input images are `14x14`. Default MNIST images are 28x28, for efficiency sake this is typically lowered to scale the images down and still have the network classify correctly (Default is 14)

- **in_channels**: The number of input channels (1 for grayscale images, 3 for RGB images). MNIST images are grayscale so this can stay at 1. (Default is 1)

- **model_output_filename**: The filename where the model will be saved after training. (Default is "mnist_cnn")

### Network Layer Configuration

The `layers` field contains a list that defines the architecture of the network. Each dictionary in the list corresponds to a layer, specifying its type and relevant parameters. Layers are assembled in the order they are listed, and the tuneable parameters of each layer must be specified. **Note** typically when assembling a network in code, you must calculate the inpute features post convolution going into things like the Dense layer, in this case you do not have to, the CNN class constructor will handle those calculations automatically, so you just need to specify tuneable parameters for the layer.

#### Example Layer Assembly

```json
"layers": [
    {
        "type": "conv",
        "out_channels": 8,
        "filter_size": 3,
        "stride": 1,
        "padding": 0
    },
    {
        "type": "relu"
    },
    {
        "type": "maxpool",
        "filter_size": 2,
        "stride": 2
    },
    {
        "type": "conv",
        "out_channels": 16,
        "filter_size": 3,
        "stride": 1,
        "padding": 0
    },
    {
        "type": "relu"
    },
    {
        "type": "maxpool",
        "filter_size": 2,
        "stride": 2
    },
    {
        "type": "flatten"
    },
    {
        "type": "dense",
        "out_features": 128
    },
    {
        "type": "relu"
    },
    {
        "type": "dense",
        "out_features": 10
    },
    {
        "type": "softmax"
    }
]

```

### Description of Layer Types

- conv: Convolutional layer. Takes parameters like out_channels (number of filters), filter_size (kernel size), stride, and padding. Convolutional layers extract features from the input by applying filters.

- relu: Rectified Linear Unit activation function. It introduces non-linearity to the network and helps prevent the vanishing gradient problem. It has no effect on input dimensions.

- maxpool: Max pooling layer. Reduces the spatial dimensions (height and width) of the input by taking the maximum value over a window (defined by filter_size) with a step size defined by stride.

- flatten: Converts the multi-dimensional output from previous layers (e.g., a 2D feature map) into a 1D vector for the fully connected layers.

- dense: Fully connected (dense) layer. in_features defines the number of input nodes, and out_features defines the number of output nodes (neurons).

- softmax: A softmax activation function is typically applied at the final layer for multi-class classification. It converts raw output scores into probabilities that sum to 1.

### How to Correctly Assemble a Network

To correctly assemble a CNN using the layers field in the configuration file, follow these guidelines:

- Start with a convolutional layer (conv): The first layer should always be a convolutional layer to extract features from the input image. Ensure you define the out_channels, filter_size, stride, and padding.

- Use activation layers like relu after any layer with learnable weights (Convolutions and Dense layers): It's common to apply a relu layer after each conv layer to introduce non-linearity.

- Pooling layers (maxpool): After one or more convolutional layers, you can add a pooling layer to reduce the dimensionality of the feature map. Make sure to adjust the filter_size and stride. Note that since pooling layers reduce dimensions of the feature maps, be careful not too have too many or there will not be enough data for your network to learn.

- Flatten layer before dense layers: Always include a flatten layer to convert 2D feature maps into 1D vectors before passing them into dense layers.

- Dense/Fully Connected layers (dense): These layers connect every input to every output. You need at least one dense layer before the final softmax layer for classification, because there are 10 possible outputs, the last dense layer should always be 10.

- End with softmax for classification: The final layer in a classification network should be a softmax layer to output probabilities for each class. The input to the last layer of neurons are called 'logits' which are meaningless for humans to interpret, this softmax activation converts these logits to probabilities, allowing us to see what the network things the input image is supposed to be and how strongly it thinks that.

### Common Mistakes

- Missing flatten layer: If you don't include a flatten layer before the dense layers, the network will not be able to transition from convolutional layers (which output 2D feature maps) to fully connected layers (which expect 1D vectors).

- Incorrect input/output dimensions: Ensure that the input size is compatible with the filter sizes and strides in convolution and pooling layers. If the filter size is too large or the stride too small, the dimensions may not match up correctly between layers.

- Too many pooling layers: Pooling layers reduce the spatial size of the feature map. Too many pooling layers may shrink the dimensions excessively, leading to loss of useful information.


For more math and theory behind the problem and to understand how the network learns to classify images, see the deep dive here: [Convolutional Neural Network - MNIST Handwritten Digits](docs/mnist_cnn.md)

---

# Planned implementations
- MNIST hand written numbers -> CNN
- k-means clustering showing points moving towards their cluster's means
- random forest visuals, showing multiple trees in ensemble converging on decision
- knn visuals shows clusters form
- attention model predictions