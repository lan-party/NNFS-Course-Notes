# Chapter 2 - A Layer of Neurons

inputs = [1, 2, 3, 4]
weights = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [-0.1, -0.2, -0.3, -0.4]]
biases = [5, 1, 0.25]

outputs = [inputs[0]*weights[0][0] + inputs[1]*weights[0][1] + inputs[2]*weights[0][2] + inputs[3]*weights[0][3] + biases[0],
           inputs[0]*weights[1][0] + inputs[1]*weights[1][1] + inputs[2]*weights[1][2] + inputs[3]*weights[1][3] + biases[1],
           inputs[0]*weights[2][0] + inputs[1]*weights[2][1] + inputs[2]*weights[2][2] + inputs[3]*weights[2][3] + biases[2]]

print(outputs)



outputs = []
for a in range(0, len(weights)):
    output = 0
    for b in range(0, len(weights[a])):
        output += inputs[b]*weights[a][b]
    output += biases[a]
    outputs.append(output)

print(outputs)
