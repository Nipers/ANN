# -*- coding: utf-8 -*-

from numpy.core.fromnumeric import mean
import torch
from torch import nn
from torch.nn.parameter import Parameter
class BatchNorm2d(nn.Module):
	# TODO START
	# Reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
	# Reference: https://blog.csdn.net/qq_39208832/article/details/117930625?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link
	def __init__(self, num_features, eps=1e-5, momentum=0.1):
		super(BatchNorm2d, self).__init__()
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
		# input: [batch_size, num_feature_map, height, width]
		mean = self.running_mean
		var = self.running_var
		if self.training:
			mean = input.mean([0, 2, 3])
			var = input.var([0, 2, 3], unbiased=False)
			with torch.no_grad():# Update running_mean and running_var
				self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean 
				self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
		input = (input - mean[None, ..., None, None]) / torch.sqrt(var[None, ..., None, None] + self.eps)
		input = input * self.weight[None, ..., None, None] + self.bias[None, ..., None, None]
		return input
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# print(input.get_device())
		# input: [batch_size, num_feature_map, height, width]
		keep_prob = 1 - self.p
		batch_size, num_feature_map, height, width = input.shape
		if self.training:
			return (torch.rand((batch_size, 1, height, width)) < keep_prob).float().to(input.get_device()) * input  / keep_prob
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5, use_norm=True, height=32, weight=32):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		kernel0 = 5
		kernel1 = 3
		channel0 = 100
		channel1 = 50
		if use_norm:
			self.layers = nn.Sequential(
				nn.Conv2d(in_channels=3, out_channels=channel0, kernel_size=kernel0),
				BatchNorm2d(channel0),
				nn.ReLU(),
				Dropout(drop_rate),
				nn.MaxPool2d(2),
				nn.Conv2d(in_channels=channel0, out_channels=channel1, kernel_size=kernel1),
				BatchNorm2d(channel1),
				nn.ReLU(),
				Dropout(drop_rate),
				nn.MaxPool2d(2)
			)
		else:
			self.layers = nn.Sequential(
				nn.Conv2d(in_channels=3, out_channels=channel0, kernel_size=kernel0),
				nn.ReLU(),
				Dropout(drop_rate),
				nn.MaxPool2d(2),
				nn.Conv2d(in_channels=channel0, out_channels=channel1, kernel_size=kernel1),
				nn.ReLU(),
				Dropout(drop_rate),
				nn.MaxPool2d(2)
			)
		outchannel = channel1*(((height + 1 - kernel0) // 2 + 1 - kernel1)//2) * (((weight + 1 - kernel0) // 2 + 1 - kernel1) // 2)
		self.linear = nn.Linear(outchannel, 10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		# the 10-class prediction output is named as "logits"
		logits = self.linear(self.layers(x).view(x.size(0), -1))
		y = y.long()
		# TODO END
		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
