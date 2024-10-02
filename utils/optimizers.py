import numpy as np

from utils.interfaces import LearnableLayer

class AdamOptimizer:
    """
    Adam optimization combines RMSprop and SGD with Momentum optimizations

    Regular SGD: w_(t+1) = w_t - lr * dL/dw
    Update weight dy decrementing the learning rate times the gradient of the current weight

    SGD Momentum: each time step model as position (w) and velocity (v)
    velocity changes with a momentum parameter (rho) and the learning rate (lr)
    momentum value characterizes resistance of velocity to change, lr characterizes gradient's influence
    think of weight as a particle moving through parameters, momentum is mass of that particle, and gradient is impulse that alters movement
    - w_(t+1) = w_t + v_(t+1)
    - v_(t+1) = rho * v_t - lr * dL/dw_t
    Goal: more adaptable to loss landscapes, jump small local minimum, and smooth out stochasticity of motion of weights
    
    AdaGrad (decaying learning rate method): 
    Issue with normal decaying lr is it scales down *assuming* it's getting closer to target minimum, but it might not be.
    However AdaGrad scales each parameter differently according to how much *that* parameter has changed in training,
    evaluated as the sum of squared prior gradients. Idea is that if a param changed a lot, it must have made lots of progress to the target
    Issue here is that it can decrease lr in response to loss landscape, but cannot increase

    RMSprop (fixes issue of AdaGrad so it can decrease OR increase lr when necessary)
    uses discount param Beta, indicating how much of previous velocity term is remembered.
    This means when large gradient found, lr is scaled down, when small grad encountered, lr scaled up.
    Get the benefits of decaying lr methods, without the risk of permanently decayed rate.
    - v_(t+1) = Beta * v_t - (1 - Beta) * (lr * dL/dw_t)^2
    - w_(t+1) = w_t - (lr * dL/dw_t) / (eps + sqrt(v_(t+1)))
    where eps -> epsilon, really small number to avoid dividing by 0

    Finally, Adam (adaptive moment estimation; get out of annoying local minima) 
    similar to rms prop, but gradient jump is parallel to vector m.
    we can think of this like a velocity term since it mirrors the velocity term in classical momentum
    lr is adjusted according to squared magnitudes of recent gradients,
    AND velocity term used to smooth like momentum does.
    - w_(t+1) = w_t - (lr * mhat_(t+1)) / (eps + sqrt(vhat_(t+1)))
    - v_(t+1) = Beta2 * v_t - (1 - Beta2) * (lr * dL/dw_t)^2 -> same from rmsprop
    - m_(t+1) = Beta1 * m_t + (1 - Beta1) * (lr * dL/dw_t)
    unbiased velocity and momentum:
    mhat_(t+1) = m_(t+1) / ((1 - Beta1^(t+1)))
    vhat_(t+1) = v_(t+1) / ((1 - Beta2^(t+1)))
    """

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step for bias correction

    def init_params(self, layers):
        # initialize lists to hold arrays of momentum and velocity estimates for each learnable (has wights and biases) layer
        self.m_w, self.v_w, self.m_b, self.v_b = [], [], [], []
        self.learnable_layers = [layer for layer in layers if isinstance(layer, LearnableLayer)]
        for layer in self.learnable_layers:
            self.m_w.append(np.zeros_like(layer.weights))
            self.v_w.append(np.zeros_like(layer.weights))
            self.m_b.append(np.zeros_like(layer.biases))
            self.v_b.append(np.zeros_like(layer.biases))

    def update(self):
        self.t += 1

        for i, layer in enumerate(self.learnable_layers):
            # first moment estimate for weights and biases
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * layer.grad_W
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.grad_b

            # second moment estimate update
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * layer.grad_W**2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * layer.grad_b**2

            # bias corrected first moment
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)

            # bias corrected second moment
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # step
            layer.weights -= (self.lr * m_w_hat) / (np.sqrt(v_w_hat) + self.epsilon)
            layer.biases -= (self.lr * m_b_hat) / (np.sqrt(v_b_hat) + self.epsilon)