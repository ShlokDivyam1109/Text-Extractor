import math
import numpy as np

class Neuron:
    def __init__(self, activation_function='ReLu', inputs=[]):
        self.inputs = inputs
        self.no_of_inputs = len(inputs)
        self.weights = np.ones(self.no_of_inputs)
        self.bias = 0
        self.F = Functions()

        if activation_function == 'Sigmoid':
            self.activation_function = self.F.Sigmoid
        elif activation_function == 'identity':
            self.activation_function = self.F.identity
        else:
            self.activation_function = self.F.ReLu

    def compute(self):
        try:
            x = np.array([n.F.val for n in self.inputs])
            if self.activation_function:
                self.activation_function(x, self.weights, self.bias)
            else:
                self.F.val = x
        except:
            self.F.val = self.inputs

class Functions:
    def identity(self, x, weights, bias):
        self.val = x

    def ReLu(self, x, weights, bias):
        z = np.dot(x, weights) + bias
        self.z = z
        self.val = max(0, z)
        self.d = 1 if z > 0 else 0

    def Sigmoid(self, x, weights, bias):
        z = np.dot(x, weights) + bias
        self.z = z
        self.val = 1 / (1 + math.exp(-z))
        self.d = self.val * (1 - self.val)

    def SoftMax(self, inputs):
        inputs = np.array(inputs)
        m = np.max(inputs)
        exp_vals = np.exp(inputs - m)
        self.val = exp_vals / np.sum(exp_vals)

    def cross_entropy(self, probs, true_class):
        return -math.log(probs[true_class])

class Neural_Network:
    def __init__(self, ip_len, n_hidden_layers, m_nodes_each_layer, op_len, activation_function, learning_rate):
        self.learning_rate = learning_rate
        self.layers = []

        input_layer = []
        for _ in range(ip_len):
            n = Neuron('identity', [])
            n.F.val = 0
            input_layer.append(n)
        self.layers.append(input_layer)

        for _ in range(n_hidden_layers):
            layer = []
            for _ in range(m_nodes_each_layer):
                layer.append(Neuron(activation_function, self.layers[-1]))
            self.layers.append(layer)

        output_layer = []
        for _ in range(op_len):
            if op_len == 1:
                output_layer.append(Neuron('Sigmoid', self.layers[-1]))
            else:
                output_layer.append(Neuron('identity', self.layers[-1]))
        self.layers.append(output_layer)

        self.F = Functions()

    def forward_pass(self, inputs):
        for i, val in enumerate(inputs):
            self.layers[0][i].F.val = val

        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.compute()

        if len(self.layers[-1]) > 1:
            logits = [n.F.val for n in self.layers[-1]]
            self.F.SoftMax(logits)
            return self.F.val

        return self.layers[-1][0].F.val

    def back_propagation(self, true_val):

        output_layer = self.layers[-1]

        if len(output_layer) == 1:
            neuron = output_layer[0]
            dz = neuron.F.val - true_val
            for i in range(len(neuron.weights)):
                neuron.weights[i] -= self.learning_rate * dz * neuron.inputs[i].F.val
            neuron.bias -= self.learning_rate * dz

        else:
            probs = self.F.val
            for j, neuron in enumerate(output_layer):
                dz = probs[j] - (1 if j == true_val else 0)
                for i in range(len(neuron.weights)):
                    neuron.weights[i] -= self.learning_rate * dz * neuron.inputs[i].F.val
                neuron.bias -= self.learning_rate * dz

        for l in range(len(self.layers)-2, 0, -1):
            layer = self.layers[l]
            next_layer = self.layers[l+1]

            for j, neuron in enumerate(layer):
                dA = 0
                for next_neuron in next_layer:
                    dA += next_neuron.weights[j] * (next_neuron.F.val)
                dz = dA * neuron.F.d
                for i in range(len(neuron.weights)):
                    neuron.weights[i] -= self.learning_rate * dz * neuron.inputs[i].F.val
                neuron.bias -= self.learning_rate * dz
