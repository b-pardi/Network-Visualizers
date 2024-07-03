import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    def sigmoid(self, x):
        # activation function
        return 1 / (1 + np.exp(-x))
    
    def d_sigmoid(self, x):
        # derivative of activation for back prop
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def forward(self, X):
        # go through network from input to output and return output neuron value
        # raw input to hidden layer, dots input values with weights and adds each hidden neurons' biass
        # X will b size m,2 where m is the number of training examples, and 2 for input features (0 or 1)
        # for xor, m is 4.

        self.z1 = np.dot(X, self.w1) + self.b1 # (m, 2) dot (2, 3) + (1, 3) = (m, 3)
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2 # (m, 3) dot (3, 1) + (1, 1) = (m, 1)
        self.a2 = self.sigmoid(self.z2)

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
        z2_grad = err * self.d_sigmoid(self.z2) # (m, 1)

        # del_L/del_w2 = del_L/del_z2 * del_z2/del_w2; z2 = a1*w2 + b2 -> del_z2/del_w2 = a1
        w2_grad = np.dot(self.a1.T, z2_grad) # (3, m) dot (m, 1) = (3, 1)

        # del_L/del_b2 = sum_i:m( del_L/del_z2(i) * del_z2(i) / del_b2 );
        # z_2 = a_1 * w_2 + b_2; -> del_z2(i) / del_b2 = 1
        b2_grad  = np.sum(z2_grad, axis=0, keepdims=True) # bias influences directly and equally for all examples

        # del_L/del_a1 = del_L/del_z2 * del_z2/del_a1; z2 = a1*w2 + b2 -> del_z2/del_a1 = w2
        a1_grad = (z2_grad * self.w2.T) # (m, 1) dot (1, 3) = (m, 3)

        # del_L/del_z1 = del_L/del_a1 * del_a1/del_z1; a1 = sig(z1) -> del_a1/del_z1 = d_sig(z1)
        z1_grad = a1_grad * self.d_sigmoid(self.z1) # (3, m) dot (m, 3) = (m, 3)

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
                self.update_visualization(X, epoch)
                self.update_error_plot(epoch)
            progress_bar.set_postfix(epoch=epoch, loss=loss)

    def initialize_visualization(self):
        self.fig = plt.figure(figsize=(15, 6))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.5, 1])  # grid spec layout, make network plot 1.5x width of loss
        self.ax_nn = self.fig.add_subplot(gs[0, 0])
        self.ax_loss = self.fig.add_subplot(gs[0, 1])
        self.ax_nn.axis('off')
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')

        # Set the aspect ratio to be equal
        self.ax_nn.set_aspect('equal', adjustable='box')

        # Define neuron positions for a 3-layer network like the image
        self.input_neurons = [(0, 1.75), (0, 3.25)]
        self.hidden_neurons = [(2, 1), (2, 2.5), (2, 4)]
        self.output_neurons = [(4, 2.5)]

        # Plot neurons
        for neuron in self.input_neurons + self.hidden_neurons + self.output_neurons:
            self.ax_nn.add_patch(plt.Circle(neuron, 0.1, color='black'))

        # Define connections
        self.connections = [
            (self.input_neurons[0], self.hidden_neurons[0]),
            (self.input_neurons[0], self.hidden_neurons[1]),
            (self.input_neurons[0], self.hidden_neurons[2]),
            (self.input_neurons[1], self.hidden_neurons[0]),
            (self.input_neurons[1], self.hidden_neurons[1]),
            (self.input_neurons[1], self.hidden_neurons[2]),
            (self.hidden_neurons[0], self.output_neurons[0]),
            (self.hidden_neurons[1], self.output_neurons[0]),
            (self.hidden_neurons[2], self.output_neurons[0])
        ]

        # Plot connections
        for conn in self.connections:
            self.ax_nn.plot([conn[0][0], conn[1][0]], [conn[0][1], conn[1][1]], 'k-', linewidth=1, alpha=0.4)

        plt.ion()  # Interactive mode on
        self.fig.canvas.mpl_connect('key_press_event', self.on_quit_key)
        plt.tight_layout()
        plt.show()

    def update_visualization(self, X, epoch):
        self.forward(X)
        self.ax_nn.clear()
        self.ax_nn.axis('off')

        # Plot neurons with updated values
        for i, neuron in enumerate(self.input_neurons):
            self.ax_nn.add_patch(plt.Circle(neuron, 0.1, edgecolor='black', fill=False))
            for j in range(X.shape[0]): # go through all (4) x inputs
                self.ax_nn.text(neuron[0] - 0.4, neuron[1] - 0.1 + 0.12*j, f'{X[j, i]:.2f}', ha='left', va='top')

        for i, neuron in enumerate(self.hidden_neurons):
            self.ax_nn.add_patch(plt.Circle(neuron, 0.1, color='black', alpha=self.a1[0, i]))
            self.ax_nn.text(neuron[0], neuron[1] + 0.4, f'Z: {self.z1[0, i]:.4f}', ha='center', va='top')
            self.ax_nn.text(neuron[0], neuron[1] + 0.25, f'A: {self.a1[0, i]:.4f}', ha='center', va='top')
            self.ax_nn.text(neuron[0], neuron[1] - 0.15, f'B: {self.b1[0, i]:.4f}', ha='center', va='top')

        for i, neuron in enumerate(self.output_neurons):
            self.ax_nn.add_patch(plt.Circle(neuron, 0.1, color='black', alpha=self.a2[0, i]))
            self.ax_nn.text(neuron[0], neuron[1] + 0.4, f'Z: {self.z2[0, i]:.4f}', ha='center', va='top')
            self.ax_nn.text(neuron[0], neuron[1] + 0.25, f'A: {self.a2[0, i]:.4f}', ha='center', va='top')
            self.ax_nn.text(neuron[0], neuron[1] - 0.15, f'B: {self.b2[0, i]:.4f}', ha='center', va='top')

        # Plot connections
        for conn in self.connections:
            self.ax_nn.plot([conn[0][0], conn[1][0]], [conn[0][1], conn[1][1]], 'k-', linewidth=1, alpha=0.5)

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
        plt.pause(0.01)

    def on_quit_key(self, e):
        if e.key == 'q':
            self.stop_flag = True

def mse(y, y_hat):
    return np.mean(np.square(y - y_hat))

if __name__ == '__main__':
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

    net = SmallNet(3, 100)
    net.initialize_visualization()
    net.train(X, y)

    y_pred = net.forward(X)
    print(f"Predicted output:\n {y_pred}")