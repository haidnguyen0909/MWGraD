import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
#from model_lenet import RegressionModel



class RegressionModel(torch.nn.Module):
	def __init__(self, n_tasks):
		super(RegressionModel, self).__init__()
		self.n_tasks = n_tasks
		self.conv1 = nn.Conv2d(1, 10, 9, 1)
		self.conv2 = nn.Conv2d(10, 20, 5, 1)
		self.fc1 = nn.Linear(5*5*20, 50)
		self.encoder = nn.Sequential(self.conv1, self.conv2, self.fc1)

		for i in range(self.n_tasks):
			layer = nn.Linear(50, 10)
			setattr(self, "task_{}".format(i+1), layer)

	def forward(self, x, i=None):
		x = self.encode(x)
		return self.decode(x, i)

	def encode(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 5*5*20)
		x = F.relu(self.fc1(x))
		return x

	def decode(self, x, i=None):
		if i is not None:
			layer_i = getattr(self, "task_{}".format(i))
			return layer_i(x)
		outs =[]
		for i in range(self.n_tasks):
			layer = getattr(self, "task_{}".format(i+1))
			outs.append(layer(x))
		return outs

	def get_shared_parameters(self):
		params = [
			{"params":self.encoder.parameters(), "lr_mult":1},
		]
		return params

	def get_classifier_parameters(self):
		params =[]
		for i in range(self.n_tasks):
			layer = getattr(self, "task_{}".format(i+1))
			params.append({"params":layer.parameters()})
		return params








