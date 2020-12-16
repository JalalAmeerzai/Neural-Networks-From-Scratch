import numpy as np

#Batch three inputs consisting of four features each
layer1_inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

#Weights for each neuron on the current layer in form of list inside a list
layer1_weights = [[0.2, 0.8, -0.5, 1.0],
			[0.5, -0.91, 0.26, -0.5],
			[-0.26, -0.27, 0.17, 0.87]]

#Each Neuron's bias on current layer
layer1_biases = [2, 3, 0.5]


#output function for current layer using dot product of Numpy for matrix multiplication and T is used to calculate transpose of matrix to correct the matrix multiplication constraints
layer1_output = np.dot(layer1_inputs, np.array(layer1_weights).T) + layer1_biases

#display output of current layer
print(layer1_output)



#Weights for each neuron on next layer in form of list inside a list
layer2_weights = [[0.1, -0.14, 0.5],
			[-0.5, 0.12, -0.33],
			[-0.44, 0.73, -0.13]]

#Each Neuron's bias on next layer
layer2_biases = [-1, 2, -0.5]

#output function for next layer using dot product of Numpy for matrix multiplication and T is used to calculate transpose of matrix to correct the matrix multiplication constraints
layer2_output = np.dot(layer1_output, np.array(layer2_weights).T) + layer2_biases

#display output of next layer
print(layer2_output)