# Enforce an interface to ensure all derived classes have forward and backward methods
class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError