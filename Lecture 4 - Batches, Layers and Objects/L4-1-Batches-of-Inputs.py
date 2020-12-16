import numpy as np

#Batch three inputs consisting of four features each
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

#Weights for each neuron on the current layer in form of list inside a list
weights = [[0.2, 0.8, -0.5, 1.0],
			[0.5, -0.91, 0.26, -0.5],
			[-0.26, -0.27, 0.17, 0.87]]

#Each Neuron's bias on current layer
biases = [2, 3, 0.5]


#output function for current layer using dot product of Numpy for matrix multiplication and T is used to calculate transpose of matrix to correct the matrix multiplication constraints
output = np.dot(inputs, np.array(weights).T) + biases

#display output of current layer
print(output) 