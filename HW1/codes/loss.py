from __future__ import division
import numpy as np
from numpy.lib.function_base import diff


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        normed_diff = np.linalg.norm(target - input, axis = 1)
        return (np.power(normed_diff, 2)).mean(axis=0).sum() / 2.0

    def backward(self, input, target):
        # TODO START
        return target - input
        # return (target - input) / input.shape[0]
        # TODO END
    def __str__(self) -> str:
        return "E"


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name
        self.loss = np.zeros(1, dtype="f")

    def forward(self, input, target):
        # TODO START
        exp = np.exp(input - np.max(input))
        self.soft = exp / np.expand_dims(np.sum(exp, 1),-1)
        return -np.mean(np.log(self.soft) * target)
        # TODO END

    def backward(self, input, target):
		# TODO START
        return target - self.soft
		# return (target - input) / input.shape[0]
        # TODO END
    def __str__(self) -> str:
        return "S"


class HingeLoss(object):
	def __init__(self, name, margin=5):
		self.name = name
		self.margin = margin

	def forward(self, input, target):
        # TODO START
		return np.mean(np.sum((target == 0) * np.maximum(0, self.margin - input[target == 1].reshape(-1, 1) + input), axis=1))
        # TODO END

	def backward(self, input, target):
        # TODO START
		'''Your codes here'''
		grad = np.zeros_like(input)
		grad[(target == 0) & (self.margin - input[target == 1].reshape(-1, 1) + input > 0)] = -1
		grad[(target == 1)] = -np.sum(grad, axis=1)
		return grad
		# return grad / input.shape[0]
        # TODO END

	def __str__(self) -> str:
		return "H"
