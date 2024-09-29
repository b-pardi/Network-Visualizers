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

2. Install the required dependencies (virtual environment recommended):
    `python -m venv venv`
    `.\venv\Scripts\activate`
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

`python main.py --xor -hn 3 -ner 200`

**Note** It is most useful to run the XOR network with 2 or 3 hidden neurons, as those allow you to see the feature space mapping of the input points. Beyond 3 dimensions it is not feasible for us to visualize, the feature space plot is omitted. However the network will still run and train and other plots show.

## Currently Implemented Network Backgrounds:
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

![XOR is not linearly separable](imgs/xor_problem/xor_1.png)

You can see in the above plot, that there is no way to draw a single line to separate the two categories.

However if we draw two lines we can separate categories effectively
![XOR outputs can be separated with 2 lines](imgs/xor_problem/xor_2.png)

*Conceptually speaking* the equivalent to drawing two lines is to use two hidden neurons to introduce non linear relationships.

However, we are not technically drawing 2 separate lines, rather having the network transform points to a new 2D space (feature space) where the points *become* linearly separable, and then interpreting this linear separation as it relates to our input space to make a decision (and to visualize the decision boundaries). This mapping is created from the non-linearities of the activation functions. When we interpret the linear decision boundary that was found in this new 2D space as it relates to our original input space, the boundary is now curved and *resembles* a 2nd degree polynomial, however it is technically a linear piecewise function found in the feature space, where each neuron contributes a piece of the piecewise function. This curved decision boundary is better demonstrated in the intermediate training steps as it starts to figure out the solution, where we can see an accurate boundary *starting* to form.

![Solution is not technically 2 lines, rather one 2nd degree polynomial line](imgs/xor_problem/xor_3.png)

In the figure below we can see the plot showing the network has found a solution and is interpreted in our input space, and how it relates to the feature space where we found a linearly separable solution. 

![Input space vs 2D feature space](imgs/xor_problem/xor_4.png)

**Note** 2 of the True outputs (pink) have been transformed such that they essentially overlap eachother.

This visualization works as we are transforming our input space from 2D to another space still in 2D. 

If we used 3 hidden neurons, we'd be transforming the points into 3D and find a plane that separates the points, then interpreting that transformation back in our original 2D input space for an output of the decision. This pattern extends s.t. n hidden neurons will map points to n dimensions.

We can see how 3 hidden neurons gives a 3D feature space below

![Input space vs 3D feature space](imgs/xor_problem/xor_5.png)

---
## Planned implementations
- MNIST hand written numbers
- attention model predictions

## Contributing
Feel free to contribute by forking the repository, making changes, and submitting a pull request.
