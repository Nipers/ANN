from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        diff = np.linalg.norm(target - input, axis = 1)
        return np.sum(diff**2) / 2
        # TODO END

    def backward(self, input, target):
		# TODO START
        return target - input
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        return -np.mean(np.log(self.softmax(input))*target)
        # TODO END

    def backward(self, input, target):
        # TODO START
        return target - input
        # TODO END
    def softmax(self, x):
        x -= np.max(x)
        exp = np.exp(x)
        return exp / np.expand_dims(np.sum(exp, axis=1),-1)


class HingeLoss(object):
	def __init__(self, name, margin=5):
		self.name = name
		self.margin = margin

	def forward(self, input, target):
        # TODO START
		return np.mean(np.sum((target == 0)* np.maximum(0, self.margin - input[target == 1].reshape(-1, 1) + input), axis=1))
        # TODO END

	def backward(self, input, target):
        # TODO START
		'''Your codes here'''
		pass
        # TODO END

