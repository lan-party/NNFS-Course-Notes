# Chapter 5 - Categorical Cross-Entropy Loss
import numpy as np
import math

b = 5.2
print(np.log(b))
print(math.e ** np.log(b))


softmax_output = [0.2, 0.1, 0.7]
print(np.argmax(softmax_output))

target_output = [1, 0, 0]
target_output = np.eye(len(softmax_output), k=np.argmax(softmax_output))[0] # np.eye can be used to generate a one-hot array
print(target_output)

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

print(loss)

print(-math.log(0.4)) # Lower confidence turns to higher loss
print(-math.log(0.5))
print(-math.log(0.8))


softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.6, 0.3],
                            [0.2, 0.2, 0.6]])
class_targets = [0, 1, 1]

print(softmax_outputs[[0, 1, 2], class_targets])
print(softmax_outputs[range(len(softmax_outputs)), class_targets])
neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
print(neg_log)
average_loss = np.mean(neg_log)
print(average_loss)
