### Convolutional Neural Network - Mnist Handwritten Digits
**THIS IS A WORK IN PROGRESS**

Currently the network functions and is able to run the training loop, and run inference on images, visuals are not yet ready.

Here is a list of steps I outlined for this part of project, and the status of their completion, which should indicate the progress.
- [x] Code to download, check hash, and prep data to send to cnn
- [x] Define class structure and base layers, plan and organize files
- [x] Implement Conv2D layer
- [x] Xavier/Glorot weight initializations
- [x] Tests for Conv2D forward and backward
- [x] Implement ReLU and Softmax activation layers
- [x] Tests for activation layers
- [x] Implement helping layers (flatten and pooling)
- [x] Tests for helping layers
- [x] implement dense layer and derive gradients
- [x] tests for dense layer
- [x] implement CNN class training loop and layer formation
- [x] save resulting model
- [x] report more metrics while training and post training
- [x] add validation to training loop
- [x] batch images instead of updating weights on all
- [x] stochastic gradient descent instead of full GD
- [x] implement the adam optimizer (or similar)
- [x] stratified splitting of train/validation data
- [x] option to scale images down
- [x] option to reduce dataset size
- [x] config file for data loading and processing
- [x] config file for network parameters
- [x] ability to assemble network with config file (layer parameters that typically require manual calculation based on input size and tunable parameters are now automatically calculated)
- [x] args to specify if training or inferring
- [x] inference ability to show pretrained model worked (plot 4 test numbers and show the true and predicted labels)
- [x] args/ability to use custom model config file for training
- [x] args/ability to use a user's trained model for training (reads model filename from user config file)
- [x] ability to load pretrained model
- [ ] setup pygame visuals backend
- [ ] visualize optimizer (what the lr is set to currently)
- [ ] Gradient Flow Visualization
- [ ] Weight Distribution Over Time
- [ ] Activation Distribution Across Layers
- [ ] Feature Map Evolution
- [ ] Prediction Confidence Heatmap
- [ ] Feature Space Visualization (PCA/LDA)
- [ ] Layer-by-Layer Output Visualization
- [ ] filter pass live visual for sample image in training with weight and pixel values on top of filter/image
- [ ] live confusion matrix on validation images
- [ ] loop that shows images in dataset and heatmap of neurons' activations in all layers showing connections to which neurons are determining outputs for each image (can be done on trained model or validation images)
- [ ] other visuals?

**Convolution Forward Pass**

$$
O_k^i(y,x) = \sum_{c=0}^{C_in - 1} \sum_{p=0}^{F - 1} \sum_{q=0}^{F - 1} X_c^i(y \cdot S + p, x \cdot S + q) \cdot W_c^k(p,q) + b^k
$$

Where:
- $O_k^i(y,x)$ is the output at position $(y,x)$ for image $i$ and filter $k$.
- $X_c^i$ is the input data for image $i$ and input channel $c$.
- $W_c^k$ is the filter weights for filter $k$ and input channel $c$.
- $b^k$ is the bias for filter $k$.
- $F$ is the filter size.
- $S$ is the stride.
- $C_{in}$ is the number of input channels.

**Convolution Backward Pass**
The various gradient formulas arise from applying chain rule to the convolution operation above.

Gradient w.r.t the bias term

$$
\frac{\partial L}{\partial b^k} = \sum_i \sum_y \sum_x \frac{\partial L}{\partial O_i^k(y,x)} \cdot \frac{\partial O_i^k(y,x)}{\partial b^k} = \sum_i \sum_y \sum_x \frac{\partial L}{\partial O_i^k(y,x)}
$$


**Gradient w.r.t the weights**
The weights of each filter are involved in attaining the output through its multiplication with the output patches. So the gradient of the loss with respect to a weight $W_c^k(p,q)$ in a filter k is the sum over all patches where that weight contributed to the output, times the upstream gradient $\frac{\partial L}{\partial O_k^i(y,x)}$

$$
\frac{\partial L}{\partial W_c^k(p,q)} = \sum_i \sum_y \sum_x \frac{\partial L}{\partial O_k^i(y,x)} \cdot \frac{\partial O_k^i(y,x)}{\partial W_c^k(p,q)} = \sum_i \sum_y \sum_x \frac{\partial L}{\partial O_k^i(y,x)} \cdot X_c^i(y \cdot S + p, x \cdot S + q)
$$


**Gradient w.r.t the inputs**
Gradient of the input is found by considering how each input pixel contributes to multiple overlapping output patches. The gradient with respect to a pixel $X_c^i(m,n)$ is the sum of all gradients from output feature maps that were influenced by that pixel. Keeping in mind that the derivative of the output with respect to the input is just the weight at that corresponding position in the filter (hence the simplification on the right).

$$
\frac{\partial L}{\partial X_c^i(m,n)} = \sum_k \sum_y \sum_x \frac{\partial L}{\partial O_k^i(y,x)} \cdot \frac{\partial O_k^i(y,x)}{\partial X_c^i(m,n)} = \sum_k \sum_y \sum_x \frac{\partial L}{\partial O_k^i(y,x)} \cdot W_c^k(p,q)
$$

Where $m$ and $p$ are the current filter coordinates, also found in the above forward pass:
- $m = y \cdot S + p$
- $n = x \cdot S + q$