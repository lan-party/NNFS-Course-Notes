# Chapter 2 - A Layer of Neurons & Batch of Data
import numpy as np

inputs = [[1, 2, 3, 4],
         [2, 3, 4, 5],
         [0, 1, 2, 3],
         [1, 3, 2, 3],
         [2, 1, 2, 4],
         [3, 5, 0, 1],
         [4, 2, 1, 2]]

weights = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [-0.1, -0.2, -0.3, -0.4]]
biases = [5, 1, 0.25]


layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
print(layer1_outputs)


# Chapter 3 - Adding Layers

weights2 = [[0.4, 0.3, 0.2], [-0.1, -0.2, -0.3], [0.1, 0.2, 0.3]]
biases2 = [0.25, 1, 5]

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)


# Objects

np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 3)

layer1.forward(inputs)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
