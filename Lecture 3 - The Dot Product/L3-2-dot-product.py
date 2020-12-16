import numpy as np

#Four inputs from previous layer
inputs = [1, 2, 3, 2.5]

#Weights for each neuron on the current layer in form of list inside a list
weights = [[0.2, 0.8, -0.5, 1.0],
			[0.5, -0.91, 0.26, -0.5],
			[-0.26, -0.27, 0.17, 0.87]]

#Each Neuron's bias on current layer
biases = [2, 3, 0.5]


#output function for current layer using dot product of Numpy
output = np.dot(weights, inputs) + biases

#display output of current layer
print(output) 