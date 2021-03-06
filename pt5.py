# Chapter 3 - Activation Functions
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Step function
x = 0.2
y = 0
if x > 0:
    y = 1
print(y)

# Sigmoid
y = 1/(1 + np.exp(-x))
print(y)

# Rectified Linear
y = 0
if x > 0:
    y = x
print(y)


# Chapter 3 - ReLU
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

output = []
for a in inputs:
    output.append(max(0, a))
print(output)

output = np.maximum(0, inputs)
print(output)


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

X, y = spiral_data(100, 3)

layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)
