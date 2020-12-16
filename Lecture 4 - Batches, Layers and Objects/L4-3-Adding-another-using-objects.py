import numpy as np
import sklearn as sk



#will reset the random number state and regenerate same numbers as previous
np.random.seed(0)

#Batch of three inputs consisting of four features (no. of inputs) each
X = [[1, 2, 3, 2.5],
	 [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]



# Class definition of creating a new layer and by default will arguements of no. of inputs and no. of neurons
# inputs coming from the previous layer in this case len(X[0]) = 4 and no. of neurons in this layer (could be any number)
class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		# Will genrate random weights of shape no. inputs * neurons  within range of -1 to 1 (to keep calculations simple)
		self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
		
		# Will generate a list of all zeroes biases in shape of 1 * no. of neurons 
		self.biases = np.zeros((1, n_neurons))

	# Will produce an output for the current layer that we are in
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases



# Layer 1 is created with pre defined weights of shape(4,5), 4 is the no. of inputs from above batch 
# also creatprint(df.columns)es biases for all 5 neurons
layer1 = Layer_Dense(len(X[0]), 5)

# Forward() will produce the output of the currrent layer which then will be feed to the next layer. 
layer1.forward(X)

# same as above we created the next layer
# NOTE: no of inputs for this layer should be same as the no. of neurons from the previous layer
layer2 = Layer_Dense(len(layer1.output[0]), 2)

# This will produce out of layer 2 by taking as inputs, the outrput of layer1
layer2.forward(layer1.output)

# Will display the output result of layer2
print(layer2.output)