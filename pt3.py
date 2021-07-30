# Chapter 2 - Dot Product and Vector Addition
import numpy as np

inputs = [1, 2, 3]
weights = [4, 5, 6]

dot_product = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2]
print(dot_product)

vector_sum = [inputs[0]+weights[0], inputs[1]+weights[1], inputs[2]+weights[2]]
print(vector_sum)


# A Single Neuron with NumPy

inputs = [1, 2, 3, 4]
weights = [0.1, 0.2, 0.3, 0.4]
bias = 5

output = np.dot(weights, inputs) + bias
print(output)



# A Layer of Neurons with NumPy

inputs = [1, 2, 3, 4]
weights = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [-0.1, -0.2, -0.3, -0.4]]
biases = [5, 1, 0.25]

output = np.dot(weights, inputs) + biases
print(list(output))
