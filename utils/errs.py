import numpy as np

def mse(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))

def cce_loss(y_prob, y_true):
    """
    Categorical cross entropy loss

    L = -sum_i=1:C [ y_i * log(p_i) ]
    where:
        C is the number of classes
        y_i is the true label of that class
        p_i is the predicted probability that it is that class (from softmax output of final FC layer)
    """

    return -np.sum(y_true * np.log(y_prob))
    

def cce_loss_gradient(y_prob, y_true):
    """
    Gradient of Categorical cross entropy loss

    L = - sum_i=1:C [ y_i * log(p_i) ]
    where:
        C is the number of classes
        y_i is the true label of that class
        p_i is the predicted probability that it is that class (from softmax output of final FC layer)
    
    recall softmax(z_i) = p_i = e^(z_i) / sum_j=1:C [ e^(z_j) ]

    derivative of Loss w.r.t the logits -> dL/dz = dL/dp * dp/dz

    We can isolate and not need the summation, since y_i is one hot encoded,
    meaning only the term corresponding to the correct class contributes to loss
    For the correct class (say k) 
        
        dl/dp_k = -y_k/p_k = -1/p_k (since y_k will be 1 when it's the true class, 0 otherwise)

    Now for derivative of softmax output w.r.t logits
        case 1: i=k (derivative of softmax output for correct class k w.r.t logit of that class)
        dp_k/dz_k = p_k(1-p_k)

        case 2: i=k (derivative of softmax output for another class i w.r.t logit of class k)
        dp_k/dz_k = -p_i*p_k

    chain rule to get our loss: 
        for the correct class i = k:
        dL/dz_k = -1/p_k * p_k(1-p_k) = -(1-p_k) = p_k-1

        for the incorrect class i != k (recall dL/dp_i = 0 for i != k):
        dL/dz_i = dL/dp_i * dp_i/dz_k = -1/p_k * -p_i*p_k = p_i

    Therefore the gradient of categorical cross entropy loss with softmax activation simplifies to:
        dL/dz_i = p_i - y_i (since y_i is 0 for i != k, wrong classes)
    
    """
    dL = y_prob - y_true
    return dL