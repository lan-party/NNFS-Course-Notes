import numpy as np
import math
import nnfs
from nnfs.datasets import spiral_data

# Chapter 4 - The Softmax Activation Function

print("Euler constant: "+str(math.e))

layer_outputs = [4.8, 1.21, 2.385]
print(layer_outputs)
E = math.e

exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)
print(exp_values)

norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
print(norm_values)
print(sum(norm_values))


# Softmax NumPy

exp_values = np.exp(layer_outputs)
print(exp_values)
norm_values = exp_values / np.sum(exp_values)
print(norm_values)
print(sum(norm_values))

# Softmax batched
layer_outputs = [[4.8, 1.21, 2.385],
                 [1, 4, 3],
                 [2, 3, 9],
                 [4, 5, 2]]

exp_values = np.exp(layer_outputs)
print(exp_values)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

X,y = spiral_data(samples=100, classes=3)

layer1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
print(activation2.output)

softmax = Activation_Softmax()

softmax.forward([[1, 2, 3]])
print(softmax.output)
