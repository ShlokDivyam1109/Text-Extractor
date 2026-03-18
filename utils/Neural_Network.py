import math
import numpy as np

class Neuron:
	def __init__(self,activation_function,inputs=[]):
		self.inputs = inputs
		self.no_of_inputs = len(inputs)
		self.weights = np.array([1]*self.no_of_inputs)
		self.bias = 0
		self.F = Functions
		if activation_function == 'ReLu':
			self.activation_function = self.F.ReLu
		elif activation_function == 'Sigmoid':
			self.activation_function = self.F.Sigmoid
		elif activation_function == 'identity':
			self.activation_function = self.F.identity
		else:
			self.activation_function = self.F.SoftMax
		

	def compute(self):
		try:
			self.activation_function(map(lambda x: x.F.val,self.inputs),self.weights,self.bias)
		except AttributeError:
			self.F.val = self.inputs
		
class Functions:
	def identity(self,x,weights,bias):
		self.val = x
		
	def ReLu(self,x,weights,bias):
		self.z = sum(x*weights)+bias
		self.val = (self.z if self.z>0 or 0)
		self.d = self.z>0

	def Sigmoid(self,x,weights,bias):
		self.z = sum(x*weights)+bias
		self.val = (1/(1+math.e**(-self.z)))
		self.d = self.val*(1-self.val)
		
	def SoftMax(self,inputs,weights=None,bias=None):
		self.inputs = np.array(inputs)
		self.max = max(self.inputs)
		self.inputs = self.inputs - self.max
		self.val = np.array(e**x for x in self.inputs)
		self.den = sum(self.val)
		self.val = self.val/self.den
		self.prediction = 0
		for i in len(inputs):
			m = 0
			if self.val[i] > m:
				m = self.val
				self.prediction = i

	def cross_entropy(self,true_class):
		return -math.log(-self.inputs[true_class] + self.max + math.log(self.den))

class Neural_Network:
	def __init__(ip_len,n_hiddden_layers,m_nodes_each_layer,op_len,activation_function,learning_rate):
		assert n_hidden_layers > 0
		self.learning_rate = learning_rate
		self.ip_len = ip_len
		self.op_len = op_len
		self.layers = np.array(list() for i in range(n+2))
		for i in range(self.ip_len):
			x=Neuron('identity',0)
			self.layers[0].append(x)
		for i in range(n_hidden_layers):
			for j in range(m_nodes_each_layer):
				x = Neuron(activation_function,self.layers[i])
				self.layers[i+1].append(x)
		if self.op_len ==1:
			self.layers[-1].append(Neuron('Sigmoid',self.layers[-2]))
		else:
			self.layers[-1].append(Neuron('SoftMax',self.layers[-2]))
	
	def forward_pass(self,inputs):
		for i in range(self.ip_len):
			self.layers[0][i].inputs = inputs[i]
		for layer in self.layers:
			for neuron in layer:
				neuron.compute()
		return self.layers[-1][0].F.val
		
	def back_propagation(self,true_val):
		if self.op_len == 1:
			dW = self.layers[-1][0].inputs*(self.layers[-1][0].F.val - true_val)
			self.layers[-1][0].weights -= self.learning_rate*dW
			dB = self.layers[-1][0].F.val - true_val
			self.layers[-1][0].bias -= self.learning_rate*dB			
		else:
			dW = self.layers[-1][0].inputs*(self.layers[-1][0].F.val[true_val]-1)
			self.layers[-1][0].weights -= self.learning_rate*dW
			dB = self.layers[-1][0].F.val[true_val]-1
		for i in self.layers[1:-1]:
			for j in i:
				new_weights = []
				for w in range(j.weights):
					dz = j.F.val-(w==true_val)
					dW = dz*j.inputs[w]
					new_weights.append(j.weights[w]-dW*self.learning_rate)
				j.weights = new_weights
				j.bias -= self.learning_rate*(j.F.val-1)
	
