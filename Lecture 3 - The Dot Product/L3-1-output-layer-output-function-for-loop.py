#Four inputs from previous layer
inputs = [1, 2, 3, 2.5]

#Weights for each neuron on the current layer in form of list inside a list
weights = [[0.2, 0.8, -0.5, 1.0],
			[0.5, -0.91, 0.26, -0.5],
			[-0.26, -0.27, 0.17, 0.87]]

#Each Neuron's bias on current layer
biases = [2, 3, 0.5]


#dynamic output function for each neuron on current layer
layer_output = [] #Output of the current layer
for neuron_weights, neuron_bias in zip(weights, biases):
	neuron_output = 0 #Output of the given neuron
	for n_input, weight in zip(inputs, neuron_weights):
		neuron_output += n_input*weight
	neuron_output += neuron_bias
	layer_output.append(neuron_output)

#Display output of the current layer
print(layer_output) 