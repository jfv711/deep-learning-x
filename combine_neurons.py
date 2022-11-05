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


# create network
class network:
    def __init__(self, weights, bias):
        self.h1 = neuron(weights, bias) 
        self.h2 = neuron(weights, bias)
        self.o1 = neuron(weights, bias)
    
    # weights inputs, add bias, then use the activation function
    def relays(self, x):
        out_h1 = self.h1.relay(x)
        out_h2 = self.h2.relay(x)

        out_o1 = self.o1.relay(np.array([out_h1, out_h2]))
        return out_o1


# example 
weights = np.array([0,1])
bias = 0
network = network(weights, bias)
x = np.array([2, 3])
print(network.relays(x)) # 0.7216325609518421