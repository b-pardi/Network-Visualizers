import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.errs import mse
from utils.act import sigmoid, d_sigmoid

class SmallNet:
    """Small feed forward network designed to solve the classic XOR problem.
    XOR has two inputs (1/True and 0/False).
    hidden layer will be just 3 neurons, 
        we need more than one to find the nonlinear relationship,
        but not many as it is still a simple problem.
        2 is likely sufficient, but adding a bit more slightly unnecesary complexity for pedagogical purposes
    Output layer is just 1 neuron, whether it is True or False

    """    
    def __init__(self, h=3, num_epochs_per_refresh=100):
        self.w1 = np.random.randn(2,h) # 2 inputs connecting to 3 hidden
        self.b1 = np.zeros((1,h)) # bias for hidden neurons

        self.w2 = np.random.randn(h,1) # 3 hidden connecting to 1 output
        self.b2 = np.zeros((1, 1)) # bias for singular output neuron

        self.stop_flag = False
        self.num_epochs_per_refresh = num_epochs_per_refresh
        self.h = h
        self.input_size = 2
        self.output_size = 1
    
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
    
    def backward(self, X, y, lr):
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

        # update weights and biases, normalizing gradients and ensuring consistency across different batch sizes
        self.w2 -= lr * w2_grad / m 
        self.b2 -= lr * b2_grad / m 
        self.w1 -= lr * w1_grad / m 
        self.b1 -= lr * b1_grad / m 

    def train(self, X, y, epochs=50000, lr=0.1):
        progress_bar = tqdm(range(epochs), desc="Training", unit="epoch")
        self.losses = []
        for epoch in progress_bar:
            if self.stop_flag:
                print("Stopping training...")
                break
            self.forward(X)
            self.backward(X, y, lr)
            loss = mse(y, self.yhat)
            self.losses.append(loss)

            if epoch % self.num_epochs_per_refresh == 0:
                self.update_network_plot(X, epoch)
                self.update_error_plot(epoch)
                self.plot_decision_boundaries(X, y)
            progress_bar.set_postfix(epoch=epoch, loss=loss)

    def predict(self, X):
        probs = self.forward(X)
        return (probs > 0.5).astype(int)

    def initialize_visualization(self):
        self.fig = plt.figure(figsize=(20, 6))
        gs = self.fig.add_gridspec(1, 3, width_ratios=[2, 1, 1])  # grid spec layout, make the network plot 2x width of loss and decision boundaries

        # Network plot
        self.ax_nn = self.fig.add_subplot(gs[0, 0])
        self.ax_nn.axis('off')
        self.ax_nn.grid(False)
        
        # Loss plot
        self.ax_loss = self.fig.add_subplot(gs[0, 1])
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')

        # Decision boundaries plot
        self.ax_decision = self.fig.add_subplot(gs[0, 2])
        self.ax_decision.set_title('Decision Boundaries')

        plt.ion()  # Interactive mode on
        self.fig.canvas.mpl_connect('key_press_event', self.on_quit_key)
        plt.tight_layout(pad=2.0)
        plt.show()

    def plot_network(self):
        self.ax_nn.clear()
        self.ax_nn.axis('off')  # Ensure gridlines are off

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
        for layer in self.pos:
            for neuron in layer:
                self.ax_nn.add_patch(plt.Circle(neuron, 0.2, edgecolor='black', fill=False))

        # plot connections between neurons
        for i in range(len(self.pos) - 1):
            for n_start in self.pos[i]:
                for n_end in self.pos[i + 1]:
                    self.ax_nn.plot([n_start[0], n_end[0]], [n_start[1], n_end[1]], 'k-', linewidth=1, alpha=0.25)

        self.ax_nn.set_xlim(-0.2, 2 * len(self.layers) + 0.2)
        self.ax_nn.set_ylim(0.7, 2 * max(self.layers) + 0.3)
        print(f"Hidden neurons: {self.h}, current xlim: {self.ax_nn.get_xlim()}, current ylim: {self.ax_nn.get_ylim()}")
        self.ax_nn.set_aspect('equal', adjustable='datalim')  # Set the aspect ratio to be equal
        plt.draw()

    def update_network_plot(self, X, epoch):
        self.forward(X)
        self.plot_network()

        base_font_size = 13
        min_font_size = 2
        font_size = max(base_font_size - (self.h), min_font_size)

        # Plot neurons with updated values
        for i, layer in enumerate(self.pos):
            for j, neuron in enumerate(layer):
                if i == 0:  # plotting input layer
                    for k in range(X.shape[0]):
                        self.ax_nn.text(neuron[0] - 0.75, neuron[1] - 0.18 + 0.22 * k, f'{X[k, j]:.2f}', ha='left', va='top', fontsize=font_size)

                elif i == len(self.pos) - 1:  # output layer
                    self.ax_nn.add_patch(plt.Circle(neuron, 0.2, color='black', alpha=self.a2[0, 0]))
                    self.ax_nn.text(neuron[0], neuron[1] + 0.7, f'Z: {self.z2[0, 0]:.4f}', ha='center', va='top', fontsize=font_size)
                    self.ax_nn.text(neuron[0], neuron[1] + 0.48, f'A: {self.a2[0, 0]:.4f}', ha='center', va='top', fontsize=font_size)
                    self.ax_nn.text(neuron[0], neuron[1] - 0.35, f'B: {self.b2[0, 0]:.4f}', ha='center', va='top', fontsize=font_size)

                else:  # hidden layer
                    self.ax_nn.add_patch(plt.Circle(neuron, 0.2, color='black', alpha=self.a1[0, j]))
                    self.ax_nn.text(neuron[0], neuron[1] + 0.7, f'Z: {self.z1[0, j]:.4f}', ha='center', va='top', fontsize=font_size)
                    self.ax_nn.text(neuron[0], neuron[1] + 0.48, f'A: {self.a1[0, j]:.4f}', ha='center', va='top', fontsize=font_size)
                    self.ax_nn.text(neuron[0], neuron[1] - 0.35, f'B: {self.b1[0, j]:.4f}', ha='center', va='top', fontsize=font_size)

        self.ax_nn.set_title(f'Epoch {epoch}', pad=40)
        plt.draw()
        plt.pause(0.01)

    def update_error_plot(self, epoch):
        self.ax_loss.clear()
        self.ax_loss.plot(range(len(self.losses)), self.losses, 'r-')
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        x_padding = 0.06 * len(self.losses)
        self.ax_loss.set_xlim([-x_padding, epoch+x_padding])
        self.ax_loss.set_ylim([0, max(self.losses) + 0.1])
        plt.draw()

    def plot_decision_boundaries(self, X, y):
        self.ax_decision.clear()

        # define grid of plot to show the 4 points with some padding
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # make a mesh of points to cover whole space
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        
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
        grid_space = np.c_[xx.ravel(), yy.ravel()]
        
        # Predict class labels for each point in the grid
        Z = self.predict(grid_space)
        Z = Z.reshape(xx.shape)
        
        self.ax_decision.contourf(xx, yy, Z, alpha=0.8) # plot contours of decision boundaries, coloring region according to Z
        self.ax_decision.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o')
        self.ax_decision.set_xlabel('Input 1')
        self.ax_decision.set_ylabel('Input 2')
        self.ax_decision.set_title('Decision Boundaries')
        plt.draw()
        plt.pause(0.01)

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

    net = SmallNet(h, ner)
    net.initialize_visualization()
    net.plot_network()
    net.train(X, y)

    y_pred = net.forward(X)
    print(f"Predicted output:\n {y_pred}")



if __name__ == '__main__':
    run_xor(3, 100)