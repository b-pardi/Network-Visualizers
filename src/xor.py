import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os

from utils.errs import mse
from utils.activations import sigmoid, d_sigmoid
from utils.log_reg import LogisticRegression

# utilize more threads for efficiency
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

class SmallNet:
    """Small feed forward network designed to solve the classic XOR problem.
    XOR has two inputs (1/True and 0/False).
    hidden layer will be just 3 neurons, 
        we need more than one to find the nonlinear relationship,
        but not many as it is still a simple problem.
        2 is likely sufficient, but adding a bit more slightly unnecesary complexity for pedagogical purposes
    Output layer is just 1 neuron, whether it is True or False

    """    
    def __init__(self, h=3, epochs=50000, num_epochs_per_refresh=100):
        self.w1 = np.random.randn(2,h) # 2 inputs connecting to 3 hidden
        self.b1 = np.zeros((1,h)) # bias for hidden neurons

        self.w2 = np.random.randn(h,1) # 3 hidden connecting to 1 output
        self.b2 = np.zeros((1, 1)) # bias for singular output neuron

        self.stop_flag = False
        self.num_epochs_per_refresh = num_epochs_per_refresh
        self.h = h
        self.input_size = 2
        self.output_size = 1
        self.epochs = epochs

        # to store the gradients for visualization
        self.num_refresh_points = self.epochs // self.num_epochs_per_refresh
        self.w1_grads = np.zeros(self.num_refresh_points)
        self.w2_grads = np.zeros(self.num_refresh_points)
        self.w1_00_history = []
        self.w1_01_history = []
        self.loss_history = []
    
    def forward(self, X):
        # go through network from input to output and return output neuron value
        # raw input to hidden layer, dots input values with weights and adds each hidden neurons' biass
        # X will b size m,2 where m is the number of training examples, and 2 for input features (0 or 1)
        # for xor, m is 4.

        self.z1 = np.dot(X, self.w1) + self.b1 # (m, 2) dot (2, 3) + (1, 3) = (m, 3)
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2 # (m, 3) dot (3, 1) + (1, 1) = (m, 1)
        self.a2 = sigmoid(self.z2)

        return self.a2
    
    def backward(self, X, y, lr, epoch):
        '''
        Find the gradient of the loss function wrt each weight and bias
        update weights and biases to find a min of the loss function
        '''

        m = X.shape[0] # how many training examples we have
        # del_L/del_yhat = yhat - y; y_hat in our case is a2, activated output neuron
        # error is how far off from the truth our predicition is
        self.yhat = self.a2
        err = self.yhat - y # (m, 1)
        
        ''' finding gradients of loss wrt (backpropagation) using MSE and CHAIN RULE
        derivative of act determines sensitivity of changes in z
        when sig(z) is close to 0 or 1, d_sig(z) is close to 0 -> changes in z have small change in sig(z)
        if sig(z) is ~0.5, d_sig(z) is ~0.25, meaning changes in z have large effect on d_sig(z)
        so z_grad is the error
        ''' # in comments below explaining derivations, 'del' represents partial derivative

        # del_L/del_z2 = del_L/del_yhat * del_yhat/del_z2; yhat = d_sigma(z2)
        z2_grad = err * d_sigmoid(self.z2) # (m, 1)

        # del_L/del_w2 = del_L/del_z2 * del_z2/del_w2; z2 = a1*w2 + b2 -> del_z2/del_w2 = a1
        w2_grad = np.dot(self.a1.T, z2_grad) # (3, m) dot (m, 1) = (3, 1)

        # del_L/del_b2 = sum_i:m( del_L/del_z2(i) * del_z2(i) / del_b2 );
        # z_2 = a_1 * w_2 + b_2; -> del_z2(i) / del_b2 = 1
        b2_grad  = np.sum(z2_grad, axis=0, keepdims=True) # bias influences directly and equally for all examples

        # del_L/del_a1 = del_L/del_z2 * del_z2/del_a1; z2 = a1*w2 + b2 -> del_z2/del_a1 = w2
        a1_grad = (z2_grad * self.w2.T) # (m, 1) dot (1, 3) = (m, 3)

        # del_L/del_z1 = del_L/del_a1 * del_a1/del_z1; a1 = sig(z1) -> del_a1/del_z1 = d_sig(z1)
        z1_grad = a1_grad * d_sigmoid(self.z1) # (3, m) dot (m, 3) = (m, 3)

        # del_L/del_w1 = del_L/del_z1 * del_z1/del_w1; z1 = X * w1 + b1 -> del_z1/del_w1 = X
        w1_grad = np.dot(X.T, z1_grad) # (2, m) dot (m, 3) = (2, 3)
        
        # same explanation as b2_grad
        b1_grad = np.sum(z1_grad, axis=0, keepdims=True)

        # store gradients when preparing to update visuals
        if epoch % self.num_epochs_per_refresh == 0:
            idx = epoch // self.num_epochs_per_refresh
            self.w1_grads[idx] = np.linalg.norm(w1_grad)
            self.w2_grads[idx] = np.linalg.norm(w2_grad)
        
        # update weights and biases, normalizing gradients and ensuring consistency across different batch sizes
        self.w2 -= lr * w2_grad / m 
        self.b2 -= lr * b2_grad / m 
        self.w1 -= lr * w1_grad / m 
        self.b1 -= lr * b1_grad / m 

    def train(self, X, y, lr=0.1):
        epochs = self.epochs
        progress_bar = tqdm(range(epochs), desc="Training", unit="epoch")
        self.losses = []
        for epoch in progress_bar:
            if self.stop_flag:
                print("Stopping training...")
                break
            self.forward(X)
            self.backward(X, y, lr, epoch)
            loss = mse(y, self.yhat)
            self.losses.append(loss)

            if epoch % self.num_epochs_per_refresh == 0:
                self.update_network_plot(X, epoch)
                self.update_error_plot(epoch)
                self.update_decision_boundaries()
                
                if self.h == 2 or self.h == 3: # feature space plot works for 2 or 3 dimensions hidden neurons
                    self.plot_feature_space(y)
                
                if self.h == 2: # 3d gradient plot only works for 2 dims (needs loss surface, w1_grad and w2_grad)
                    self.plot_gradient_trajectory(X, y)

                if self.h >= 3: # gradient norms plot works for any dims, but use only for h >= 3 so we can do loss surface gradient plot for h = 2
                    grad_idx = epoch // self.num_epochs_per_refresh
                    self.plot_gradient_norms(grad_idx)

                # process all pending plot updates
                self.fig.canvas.flush_events()
                
            progress_bar.set_postfix(epoch=epoch, loss=loss)

    def predict(self, X):
        # Perform forward pass computations without modifying internal state
        z1 = np.dot(X, self.w1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = sigmoid(z2)
        return (a2 >= 0.5).astype(int)

    def initialize_visualization(self, X, y):
        self.fig = plt.figure(figsize=(16, 8))

        # different number of hidden neurons -> different plots -> different grid layouts
        if self.h == 2:
            gs = self.fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.6], height_ratios=[1,1])  # grid spec layout
            self.ax_nn = self.fig.add_subplot(gs[0, 0])
            self.ax_loss = self.fig.add_subplot(gs[1, 0])
            self.ax_decision = self.fig.add_subplot(gs[0, 1])
            self.ax_feature = self.fig.add_subplot(gs[1, 1]) # 2D feature space plot for 2 hidden neurons
            self.ax_gradient_trajectory = self.fig.add_subplot(gs[:, 2], projection='3d') # 3D gradient plot (loss surface and gradient path)

        elif self.h == 3:
            gs = self.fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.6], height_ratios=[1,1])  # grid spec layout
            self.ax_nn = self.fig.add_subplot(gs[0, 0])
            self.ax_loss = self.fig.add_subplot(gs[1, 0])
            self.ax_decision = self.fig.add_subplot(gs[0, 1])
            self.ax_feature = self.fig.add_subplot(gs[:, 2], projection='3d') # 3D feature space plot for 3 hidden neurons
            self.ax_gradient_norms = self.fig.add_subplot(gs[1, 1]) # 2D gradient norm; plot norm of weights in hidden and output layer
        
        else:
            gs = self.fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1], height_ratios=[1,1])  # grid spec layout
            self.ax_nn = self.fig.add_subplot(gs[:, 0])
            self.ax_loss = self.fig.add_subplot(gs[0, 1])
            self.ax_decision = self.fig.add_subplot(gs[1, 1])
            self.ax_gradient_norms = self.fig.add_subplot(gs[:, 2]) # 2D gradient norm; plot norm of weights in hidden and output layer

        # get color map to use for all plots (that need color maps)
        self.cmap = cm.get_cmap('cool', 2)
        self.cnorm = mcolors.Normalize(vmin=0, vmax=1)

        # Network plot
        self.init_network_plot()

        # Loss plot
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.plot([], [], 'r-') # temporary line so it can be updated
        
        # Decision boundaries plot
        self.init_decision_boundaries_plot(X, y)

        self.fig.canvas.draw()
        self.backgrounds = { # initialize backgrounds for blitting
            'nn': self.fig.canvas.copy_from_bbox(self.ax_nn.bbox),
            'loss': self.fig.canvas.copy_from_bbox(self.ax_loss.bbox),
            'decision': self.fig.canvas.copy_from_bbox(self.ax_decision.bbox)
        }

        # plots ommitted from blitting method:
        # feature space: 3D plots don't support blit and the 2d version needs to have axis limits adjustable

        plt.ion()  # Interactive mode on
        self.fig.canvas.mpl_connect('key_press_event', self.on_quit_key)
        plt.tight_layout(pad=1.0)
        plt.show()

    def init_network_plot(self):
        self.ax_nn.axis('off')
        self.ax_nn.grid(False)

        self.layers = [self.input_size, self.h, self.output_size]

        # get neuron positions
        self.pos = []
        for i, layer_size in enumerate(self.layers):
            if i == 0:
                y_neuron_pos = np.linspace(1, 2 * max(self.layers), layer_size*2)
                y_neuron_pos = y_neuron_pos[1:-1]
            elif i == len(self.layers) - 1:
                y_neuron_pos = [np.mean(np.linspace(1, 2 * max(self.layers), self.h))]  # Center vertically based on hidden neurons
            else:
                y_neuron_pos = np.linspace(1, 2 * max(self.layers), layer_size)
            x_neuron_pos = [i * 3] * layer_size
            layer_positions = list(zip(x_neuron_pos, y_neuron_pos))
            self.pos.append(layer_positions)

        # init circles for neurons
        self.layer_node_patches = []
        for layer in self.pos:
            layer_patches = []
            for neuron in layer:
                patch = plt.Circle(neuron, 0.2, edgecolor='black', fill=False)
                self.ax_nn.add_patch(patch)
                layer_patches.append(patch)
            self.layer_node_patches.append(layer_patches)

        # plot connections between neurons
        self.edge_lines = []
        for i in range(len(self.pos) - 1):
            for n_start in self.pos[i]:
                for n_end in self.pos[i + 1]:
                    line, = self.ax_nn.plot([n_start[0], n_end[0]], [n_start[1], n_end[1]], 'k-', linewidth=1, alpha=0.25)
                    self.edge_lines.append(line)

        x_buffer = 1.1
        total_width = 3 * (len(self.layers) - 1)  # Total width based on the number of layers and spacing
        self.ax_nn.set_xlim(-x_buffer, x_buffer + total_width)
        self.ax_nn.set_ylim(0.7, 2 * max(self.layers) + 0.3)
        self.ax_nn.set_aspect('equal', adjustable='datalim')  # Set the aspect ratio to be equal
        plt.draw()

    def update_network_plot(self, X, epoch):
        # restore static elements
        self.fig.canvas.restore_region(self.backgrounds['nn'])

        self.forward(X)

        base_font_size = 13
        min_font_size = 2
        font_size = max(base_font_size - (self.h), min_font_size)

        # remove previous text
        if hasattr(self, 'texts'):
            for txt in self.texts:
                txt.remove()
        self.texts = []

        # Plot neurons with updated values
        for i, layer_patches in enumerate(self.layer_node_patches):
            for j, patch in enumerate(layer_patches):
                neuron = self.pos[i][j]
                if i == 0:  # plotting input layer
                    for k in range(X.shape[0]):
                        text = self.ax_nn.text(neuron[0] - 0.9, neuron[1] - 0.3 + 0.3 * k, f'{X[k, j]:.2f}', ha='left', va='top', fontsize=font_size)
                        self.texts.append(text)

                elif i == len(self.pos) - 1:  # output layer
                    # update facecolor and transparency
                    patch.set_facecolor('black')
                    patch.set_alpha(self.a2[0, 0])
                    z_text = self.ax_nn.text(neuron[0], neuron[1] + 0.8, f'Z: {self.z2[0, 0]:.4f}', ha='center', va='top', fontsize=font_size)
                    a_text = self.ax_nn.text(neuron[0], neuron[1] + 0.48, f'A: {self.a2[0, 0]:.4f}', ha='center', va='top', fontsize=font_size)
                    b_text = self.ax_nn.text(neuron[0], neuron[1] - 0.35, f'B: {self.b2[0, 0]:.4f}', ha='center', va='top', fontsize=font_size)
                    self.texts.extend([z_text, a_text, b_text])

                else:  # hidden layer
                    patch.set_facecolor('black')
                    patch.set_alpha(self.a1[0, j]) 
                    z_text = self.ax_nn.text(neuron[0], neuron[1] + 0.8, f'Z: {self.z1[0, j]:.4f}', ha='center', va='top', fontsize=font_size)
                    a_text = self.ax_nn.text(neuron[0], neuron[1] + 0.48, f'A: {self.a1[0, j]:.4f}', ha='center', va='top', fontsize=font_size)
                    b_text = self.ax_nn.text(neuron[0], neuron[1] - 0.35, f'B: {self.b1[0, j]:.4f}', ha='center', va='top', fontsize=font_size)
                    self.texts.extend([z_text, a_text, b_text])

        self.ax_nn.set_title(f'Epoch {epoch}', pad=40)
        # Draw the updated dynamic elements
        for patch in sum(self.layer_node_patches, []):
            self.ax_nn.draw_artist(patch)
        for txt in self.texts:
            self.ax_nn.draw_artist(txt)
        self.ax_nn.draw_artist(self.ax_nn.title)

        # Blit the axes
        self.fig.canvas.blit(self.ax_nn.bbox)

    def update_error_plot(self, epoch):
        # restore static elements
        self.fig.canvas.restore_region(self.backgrounds['loss'])
        
        # update dynamic data (plotting but with blit)
        self.ax_loss.lines[0].set_data(range(len(self.losses)), self.losses)
        x_padding = 0.06 * len(self.losses)
        self.ax_loss.set_xlim([-x_padding, epoch+x_padding])
        self.ax_loss.set_ylim([0, max(self.losses) + 0.1])
        
        # instead of draw, use draw artist to only draw updated dynamic objects (like the line)
        self.ax_loss.draw_artist(self.ax_loss.lines[0])

        # blit to canvas
        self.fig.canvas.blit(self.ax_loss.bbox)

    def init_decision_boundaries_plot(self, X, y):
        self.ax_decision.set_title('Decision Boundaries')
        self.ax_decision.set_xlabel('Input 1')
        self.ax_decision.set_ylabel('Input 2')

        # define grid of plot to show the 4 points with some padding
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # define boundaries of image to match axes limits (image replaces contour plot for efficiency)
        self.extent = [x_min, x_max, y_min, y_max]

        # make a mesh of points to cover whole space
        self.xx, self.yy = np.meshgrid(np.arange(x_min, x_max, 0.04),
                            np.arange(y_min, y_max, 0.04))
        
        # flatten xx and yy meshes and stack them into 2d arr
        # each row in [] is a column vector and is a point in the input spaced
        '''
        xx:
            [[0 1 2]
            [0 1 2]
            [0 1 2]]
        yy:
            [[0 0 0]
            [1 1 1]
            [2 2 2]]
        grid_points:
            [[0 0]
            [1 0]
            [2 0]
            [0 1]
            [1 1]
            [2 1]
            [0 2]
            [1 2]
            [2 2]]
        '''
        self.grid_space = np.c_[self.xx.ravel(), self.yy.ravel()]
        print(self.grid_space.shape)

        self.ax_decision.scatter(X[:, 0], X[:, 1], c=y, cmap='cool', edgecolor='k', marker='o', zorder=10)

        # Create custom handles for the two classes
        class_0 = mpatches.Patch(color=self.cmap(self.cnorm(0)), label='False (0)')
        class_1 = mpatches.Patch(color=self.cmap(self.cnorm(1)), label='True (1)')

        self.ax_decision.legend(handles=[class_0, class_1], loc='best')

        # Initialize the contour plot (dynamic element)
        # using imshow to display decision boundaries is more efficient with blitting than contour
        Z = np.zeros_like(self.xx) 
        self.decision_img = self.ax_decision.imshow(Z, extent=self.extent, cmap=self.cmap, norm=self.cnorm, alpha=0.8, origin='lower', aspect='auto', zorder=1)

    def update_decision_boundaries(self):
        # Restore background
        self.fig.canvas.restore_region(self.backgrounds['decision'])

        # Predict class labels for each point in the grid
        Z = self.predict(self.grid_space)
        Z = Z.reshape(self.xx.shape)

        # Update the pcolormesh data
        self.decision_img.set_data(Z)

        # Draw the updated pcolormesh
        self.ax_decision.draw_artist(self.decision_img)

        # Blit the axes
        self.fig.canvas.blit(self.ax_decision.bbox)
        self.fig.canvas.flush_events()

    def plot_gradient_norms(self, grad_idx):
        self.w1_grads[grad_idx] = np.linalg.norm(self.w1)
        self.w2_grads[grad_idx] = np.linalg.norm(self.w2)
                
        # Plot gradient norms
        self.ax_gradient_norms.cla()
        self.ax_gradient_norms.plot(np.arange(grad_idx + 1) * self.num_epochs_per_refresh, self.w1_grads[:grad_idx + 1], label='w1 gradient norm', color='blue')
        self.ax_gradient_norms.plot(np.arange(grad_idx + 1) * self.num_epochs_per_refresh, self.w2_grads[:grad_idx + 1], label='w2 gradient norm', color='purple')
        self.ax_gradient_norms.set_title('Gradient Norms')
        self.ax_gradient_norms.set_xlabel('Epoch')
        self.ax_gradient_norms.set_ylabel('Gradient Norm')
        self.ax_gradient_norms.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_gradient_trajectory(self, X, y):
        """
        Update the 3D gradient plot to visualize the loss surface and the trajectory of weights.

        Parameters:
        - grad_idx (int): The current index in the gradients array, corresponding to the epoch.
        - X (numpy.ndarray): Input data of shape (m, 2), where m is the number of examples.
        - y (numpy.ndarray): True labels of shape (m, 1).

        This method performs the following steps:
        1. Defines a grid of values around the current weights w1[0,0] and w1[0,1].
        2. Computes the loss at each point on the grid, keeping other weights constant.
        3. Plots the loss surface as a 3D plot.
        4. Tracks and plots the trajectory of the selected weights over the loss surface.
        """
        # Define ranges for w1[0,0] and w1[0,1] around their current values
        delta = 5  # Adjust the range as needed
        num_points = 30  # Number of points in the grid
        w1_00 = self.w1[0, 0]
        w1_01 = self.w1[0, 1]

        x0, xf = w1_00 - delta, w1_00 + delta
        y0, yf = w1_01 - delta, w1_01 + delta

        w1_00_vals = np.linspace(x0, xf, num_points)
        w1_01_vals = np.linspace(y0, yf, num_points)
        W1_00, W1_01 = np.meshgrid(w1_00_vals, w1_01_vals)
        Loss = np.zeros_like(W1_00)

        # Compute loss at each grid point
        for i in range(W1_00.shape[0]):
            for j in range(W1_00.shape[1]):
                w1_copy = self.w1.copy()
                w1_copy[0, 0] = W1_00[i, j]
                w1_copy[0, 1] = W1_01[i, j]
                # Forward pass with modified weights
                # simulate a forward pass as to not modify the weights in the actual network
                z1 = np.dot(X, w1_copy) + self.b1
                a1 = sigmoid(z1)
                z2 = np.dot(a1, self.w2) + self.b2
                a2 = sigmoid(z2)
                loss = mse(y, a2)
                Loss[i, j] = loss

        # Clear and plot the loss surface
        self.ax_gradient_trajectory.clear()
        self.ax_gradient_trajectory.plot_surface(W1_00, W1_01, Loss, alpha=0.4, cmap='viridis')
        self.ax_gradient_trajectory.set_xlabel('Neuron 1 Weight')
        self.ax_gradient_trajectory.set_ylabel('Neuron 2 Weight')
        self.ax_gradient_trajectory.set_zlabel('Loss Surface')
        self.ax_gradient_trajectory.set_title('Loss Surface and Weight Trajectory')

        # Append current weights and loss to history
        self.w1_00_history.append(w1_00)
        self.w1_01_history.append(w1_01)
        self.loss_history.append(self.losses[-1])

        # Plot the trajectory of weights over the loss surface
        self.ax_gradient_trajectory.plot3D(self.w1_00_history, self.w1_01_history, self.loss_history, 'r.-')

        # Redraw the plot
        plt.draw()

    def plot_feature_space(self, y):
        self.ax_feature.cla()
        activations = self.a1  # self.a1 contains the activations of the first hidden layer

        # view decision boundary for feature space
        log_reg = LogisticRegression()
        log_reg.fit(activations, y)

        # Determine the number of hidden neurons
        if self.h == 2:
            # Plot in 2D
            self.ax_feature.scatter(activations[:, 0], activations[:, 1], s=100, c=y, cmap='cool')
            plt.xlabel('Activation of Neuron 1')
            plt.ylabel('Activation of Neuron 2')
            plt.title('2D Feature Space Transformed by Hidden Layer')

            # plot boundary line
            x = np.array([activations[:, 0].min() - 1, activations[:, 0].max() + 1])
            yhat = -(log_reg.weights[0] * x + log_reg.bias) / log_reg.weights[1]
            self.ax_feature.plot(x, yhat, color='black', linestyle='--')
        elif self.h == 3:
            # Plot in 3D
            self.ax_feature.scatter(activations[:, 0], activations[:, 1], activations[:, 2], s=200, c=y, cmap='cool')
            self.ax_feature.set_xlabel('Activation of Neuron 1')
            self.ax_feature.set_ylabel('Activation of Neuron 2')
            self.ax_feature.set_zlabel('Activation of Neuron 3')
            self.ax_feature.set_title('3D Feature Space Transformed by Hidden Layer')

            # plot boundary plane
            x_range = np.linspace(activations[:, 0].min() - 1, activations[:, 0].max() + 1, 10)
            y_range  = np.linspace(activations[:, 1].min() - 1, activations[:, 1].max() + 1, 10)
            xx, yy = np.meshgrid(x_range, y_range)
            zz = -(log_reg.weights[0] * xx + log_reg.weights[1] * yy + log_reg.bias) / log_reg.weights[2]
            self.ax_feature.plot_surface(xx, yy, zz, color='black', alpha=0.25)

    def on_quit_key(self, e):
        if e.key == 'q':
            self.stop_flag = True

def run_xor(h=3, ner=100):
    X = np.array([ # input data is a 2bit truth table
        [0,0], # if both are False...
        [0,1], # if first is False and second is True...
        [1,0], # if first is True and second is False...
        [1,1] # if both are True...
    ])

    y = np.array([
        [0], # then XOR gives False
        [1], # then XOR gives True
        [1], # then XOR gives True
        [0] # then XOR gives False
    ])

    net = SmallNet(h, epochs=50000, num_epochs_per_refresh=ner)
    net.initialize_visualization(X, y)
    net.train(X, y)

    y_pred = net.forward(X)
    print(f"Predicted output:\n {y_pred}")



if __name__ == '__main__':
    run_xor(3, 100)