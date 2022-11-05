# install packages
from math import e
import numpy as np

# create sigmoid function 
def sigmoid(x): 
    # calculation: f(x) = 1 / (1 + e^(-x))
    sig = np.power(e,(-x))
    return 1/(1+sig)

# create neuron 
class neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    # weights inputs, add bias, then use the activation function
    def relay(self, inputs):
        # calculation: y = f(x1 * w1 + x2 * w2 + b)
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


# example
weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n = neuron(weights, bias)

x = np.array([2, 3])       # x1 = 2, x2 = 3
print(n.relay(x))    # 0.9990889488055994