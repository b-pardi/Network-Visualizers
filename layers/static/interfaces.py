# Enforce an interface to ensure all derived classes have forward and backward methods
class Layer:
    def forward(self, input):
        raise NotImplementedError()

    def backward(self, d_out):
        raise NotImplementedError()
    
class LearnableLayer(Layer):
    def __init__(self):
        super().__init__()
        self.weights = None
        self.biases = None
        self.grad_W = None
        self.grad_b = None

    def update_params(self, optimizer):
        raise NotImplementedError()
