# Network-Visualizers

### Personal implementations of various networks without any ML libraries and visualizations for each. (Currently only XOR implemented)
---
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