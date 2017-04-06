


# create the xor dataset to test the neuron


#==========================================================================
# feed forward
#==========================================================================

# we will create a very simple neuron 2x3x1
# the neuron will take 2 inputs x1, x2
# have randomly initialized weights w11,12,13 and w21,22,23 with bias w10, w20
# for each set of inputs pass x1 and x2 to h1,h2,h3
# h1,h2,h3 will generate outputs o1,o2,o3
# the outputs are summed and used to generate a prediction
# the output prediction is based on Logistic fcn


from random import seed
from random import random
from math import exp

seed(1)

#==========================================================================
# back prop
#==========================================================================

class MLP(object):
    "neuron with 1 hidden layer"
    def __init__(self,n_inputs,n_hidden,n_outputs,l_rate,n_epoch,verbose=True):
        self.network = self.initialize_network(n_inputs,n_hidden,n_outputs)
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.n_outputs = n_outputs
        self.verbose = verbose

    # Initialize a network
    def initialize_network(self,n_inputs, n_hidden, n_outputs):
        network = list()
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
        return network

    # Calculate neuron activation for an input
    def activate(self,weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation

    # Transfer neuron activation
    def transfer(self,activation):
        return 1.0 / (1.0 + exp(-activation))

    # Forward propagate input to a network output
    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # Calculate the derivative of an neuron output
    def transfer_derivative(self,output):
        return output * (1.0 - output)

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()

            # since we count backwards, this is first if captures the i=0 weights
            # NOTE: this implementation can only support 1 hidden layer
            # i.e. in a 2,1,2 network. we start counting at i=1 len(network)-1=1
            if i != len(self.network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            # here we calculate the error between expected and predicted
            # for each neuron in the hidden layer
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            # this captures the delta w of each neuron in the hidden layer
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    # NOTE: row here expects the full dataset with target as the last col
    # Update network weights with error
    def update_weights(self, row):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += self.l_rate * neuron['delta']

    # NOTE: row here expects the full dataset with target as the last col
    # NOTE: a better implementation is X and y entries
    # Train a network for a fixed number of epochs
    def fit(self, train):
        for epoch in range(self.n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.forward_propagate(row)
                expected = [0 for i in range(self.n_outputs)]
                expected[row[-1]] = 1
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row)
            if self.verbose:
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.l_rate, sum_error))

    # Make a prediction with a network
    def predict(self, dataset):
        pred = []
        for row in dataset:
            outputs = self.forward_propagate(row)
            pred.append(outputs.index(max(outputs)))
        return pred

#==========================================================================
# from machinelearning mastry:
#==========================================================================
# http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# the following is cobbled together from the tutorial in the link above...
# quite a bit of clean up would make this a nicer model to run!
# also. some of what happens here is not completely clear. it would be worth making it much more explicit (perhaps tying the functions together with a blog post)

# from random import seed
# from random import random
# from math import exp
#
# seed(1)
#
#
# #==========================================================================
# # Initialize the network
# #==========================================================================
#
#
# # Initialize a network
# def initialize_network(n_inputs, n_hidden, n_outputs):
# 	network = list()
# 	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
# 	network.append(hidden_layer)
# 	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
# 	network.append(output_layer)
# 	return network
#
#
# # network = initialize_network(2, 1, 2)
# # for layer in network:
# # 	print(layer)
#
# #==========================================================================
# # Forward Propigation
# #==========================================================================
#
#
# # NOTE: UPDATE TO LINEAR ALGEBRA... might be cleaner / easier to understand
# # ALSO! It isn't clear how the bias will be handled moving forward... it does appear that every neuron has an extra input for a bias term! What about the input?
#
# # Calculate neuron activation for an input
# def activate(weights, inputs):
# 	activation = weights[-1]
# 	for i in range(len(weights)-1):
# 		activation += weights[i] * inputs[i]
# 	return activation
#
# # Transfer neuron activation
# def transfer(activation):
# 	return 1.0 / (1.0 + exp(-activation))
#
# # Forward propagate input to a network output
# def forward_propagate(network, row):
# 	inputs = row
# 	for layer in network:
# 		new_inputs = []
# 		for neuron in layer:
# 			activation = activate(neuron['weights'], inputs)
# 			neuron['output'] = transfer(activation)
# 			new_inputs.append(neuron['output'])
# 		inputs = new_inputs
# 	return inputs
#
# # test forward propagation
# # network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
# # 		[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
# network = initialize_network(2,1,2)
# for layer in network:
#     print layer
# row = [1, 0, None]
# output = forward_propagate(network, row)
# print(output)
#
#
# #==========================================================================
# # Back Propigation
# #==========================================================================
#
# # Calculate the derivative of an neuron output
# def transfer_derivative(output):
# 	return output * (1.0 - output)
#
# # Backpropagate error and store in neurons
# def backward_propagate_error(network, expected):
#     for i in reversed(range(len(network))):
#         layer = network[i]
#         errors = list()
#
#         # since we count backwards, this is first if captures the i=0 weights
#         # NOTE: this implementation can only support 1 hidden layer
#         # i.e. in a 2,1,2 network. we start counting at i=1 len(network)-1=1
#         if i != len(network)-1:
#             for j in range(len(layer)):
#                 error = 0.0
#                 for neuron in network[i + 1]:
#                     error += (neuron['weights'][j] * neuron['delta'])
#                 errors.append(error)
#         # here we calculate the error between expected and predicted
#         # for each neuron in the hidden layer
#         else:
#             for j in range(len(layer)):
#                 neuron = layer[j]
#                 errors.append(expected[j] - neuron['output'])
#         # this captures the delta w of each neuron in the hidden layer
#         for j in range(len(layer)):
#             neuron = layer[j]
#             neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
#
# # # test backpropagation of error
# # network = [[{'output': 0.7105668883115941,'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
# # 		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
# # expected = [0, 1]
# # backward_propagate_error(network, expected)
# # for layer in network:
# # 	print(layer)
#
#
# # Update network weights with error
# # NOTE: row here expects the full dataset with classification as the last col
# def update_weights(network, row, l_rate):
# 	for i in range(len(network)):
# 		inputs = row[:-1]
# 		if i != 0:
# 			inputs = [neuron['output'] for neuron in network[i - 1]]
# 		for neuron in network[i]:
# 			for j in range(len(inputs)):
# 				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
# 			neuron['weights'][-1] += l_rate * neuron['delta']
#
# # Train a network for a fixed number of epochs
# # NOTE: train here is the entire training set with classification as the last col. This really should be an X and y type implementation
# def train_network(network, train, l_rate, n_epoch, n_outputs):
# 	for epoch in range(n_epoch):
# 		sum_error = 0
# 		for row in train:
# 			outputs = forward_propagate(network, row)
# 			expected = [0 for i in range(n_outputs)]
# 			expected[row[-1]] = 1
# 			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
# 			backward_propagate_error(network, expected)
# 			update_weights(network, row, l_rate)
# 		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
#
# #==========================================================================
# # Predict
# #==========================================================================
# # Make a prediction with a network
# def predict(network, row):
# 	outputs = forward_propagate(network, row)
# 	return outputs.index(max(outputs))

#==========================================================================
# Test the fcn above
#==========================================================================

# # Test training backprop algorithm
# dataset = [[2.7810836,2.550537003,0],
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	[3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# 	[7.673756466,3.508563011,1]]
# n_inputs = len(dataset[0]) - 1
# n_outputs = len(set([row[-1] for row in dataset]))
# network = initialize_network(n_inputs, 2, n_outputs)
# train_network(network, dataset, 0.5, 20, n_outputs)
#
# # for layer in network:
# # 	print(layer)
#
# for row in dataset:
#     prediction = predict(network, row)
#     print('Expected=%d, Got=%d' % (row[-1], prediction))


if __name__ == '__main__':

    # Test training backprop algorithm
    dataset = [[2.7810836,2.550537003,0],
    	[1.465489372,2.362125076,0],
    	[3.396561688,4.400293529,0],
    	[1.38807019,1.850220317,0],
    	[3.06407232,3.005305973,0],
    	[7.627531214,2.759262235,1],
    	[5.332441248,2.088626775,1],
    	[6.922596716,1.77106367,1],
    	[8.675418651,-0.242068655,1],
    	[7.673756466,3.508563011,1]]

    clf = MLP(n_inputs=2,n_hidden=1,n_outputs=2,
              l_rate=0.5,n_epoch=25,verbose=True)
    clf.fit(dataset)
    pred = clf.predict(dataset)
