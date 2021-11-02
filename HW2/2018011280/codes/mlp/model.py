# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features, eps=1e-5, momentum=0.1):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum

		# Parameters		
		self.weight = Parameter(torch.ones(num_features, requires_grad=True)) # initially the weight matrix is all one
		self.bias = Parameter(torch.zeros(num_features, requires_grad=True)) # bias is zero

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))# average is zero
		self.register_buffer('running_var', torch.zeros(num_features))#variance is one
		
		# Initialize your parameter

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		batch_size = input.shape[0]
		mean = self.running_mean
		var = self.running_var
		if self.training:
			mean = torch.mean(input, dim=0)
			var = torch.var(input, dim=0) * batch_size / (batch_size - 1)
			with torch.no_grad():
				self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
				self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
		input = (input - mean) / (var + self.eps).sqrt()
		input = self.weight * input + self.bias
		return input
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		keep_prob = 1 - self.p
		if self.training:
			return (torch.rand(input.shape) < keep_prob).float().to(input.get_device()) * input  / keep_prob
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5, use_norm=True, inputsize = 32 * 32 * 3, hiddensize = 200):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		if use_norm:
			self.layers = nn.Sequential(
				nn.Linear(inputsize, hiddensize),
				BatchNorm1d(hiddensize),
				nn.ReLU(),
				Dropout(drop_rate),
				nn.Linear(hiddensize, 10),
			)
		else:
			self.layers = nn.Sequential(
				nn.Linear(inputsize, hiddensize),
				nn.ReLU(),
				Dropout(drop_rate),
				nn.Linear(hiddensize, 10),
			)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		logits = self.layers(x)
		y = y.long()
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
