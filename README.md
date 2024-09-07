# Network-Visualizers
Personal implementations of various networks without any ML libraries and visualizations for each. (Currently only XOR implemented)

## Table of Contents

- [Setup and Installation](#setup-and-installation)
- [Handling CLI Arguments](#handling-cli-arguments)
  - [Example Usage](#example-usage)
- [Currently Implemented](#currently-implemented)
  - [XOR (Exclusive OR) Problem Neural Network](#xor-exclusive-or-problem-neural-network)
- [Planned Implementations](#planned-implementations)
- [Contributing](#contributing)
---
## Setup and Installation

1. Clone the repository:
    `git clone https://github.com/yourusername/Network-Visualizers.git`
    `cd Network-Visualizers`

2. Install the required dependencies:
    `pip install -r requirements.txt`

3. Run the desired neural network visualizer:

    - **XOR Problem**:
    `python main.py --xor`

    - **MNIST Digit Recognition (Planned)**:
    `python main.py --mnist_digits`

4. To exit training early, press 'q'

## Handling CLI Arguments

The script accepts several command-line arguments to control the visualization and configuration of the networks:

- `--xor`: Run the XOR classifier
- `-hn`, `--hidden_neurons`: Number of neurons in the hidden layer (default is 3)
- `-ner`, `--num_epochs_refresh_visualizer`: Number of epochs per refreshing of visualizers (affects the speed of training; default is 100)
- `-md`, `--mnist_digits`: Run the MNIST digit classifier (planned implementation)

### Example Usage

To run the XOR neural network with a custom number of hidden neurons and visualizer refresh rate:

`python main.py --xor -hn 5 -ner 200`

## Currently implemented:
### XOR (Exclusive OR) Problem Neural Net
Simple neural network with 3 layers (1 input, 1 hidden, 1 output) to solve the XOR gate problem.
The XOR problem is a fundamental in machine learning, introducing the concept non linearly separable classification

XOR takes two binary inputs and outputs a single binary output as shown in the following truth table:
```
    a   |   b   |   y
-------------------------
    0   |   0   |   0       # if both inputs are false, output is false
    0   |   1   |   1       # if one input is false and one true, output is true
    1   |   0   |   1       # if one input is true and one false, output is true
    1   |   1   |   0       # if both inputs are true, output is false
```

Plotting the inputs and outputs of this truth table results in something like this:

![XOR is not linearly separable (Credit: Codeacademy)](https://toritris.weebly.com/uploads/1/4/1/3/14134854/6790963_orig.jpg)
You can see in the above plot, that there is no way to draw a single line to separate the two categories

However if we draw two lines we can separate categories effectively
![XOR outputs can be separated with 2 lines (Credit: Codeacademy)](https://toritris.weebly.com/uploads/1/4/1/3/14134854/2921577_orig.jpg)



---
## Planned implementations
- MNIST hand written numbers
- attention model predictions

## Contributing
Feel free to contribute by forking the repository, making changes, and submitting a pull request.
