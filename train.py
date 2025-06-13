
# This code is adapted from the implementation of MT-SGD (https://github.com/VietHoang1512/MT-SGD).


import argparse
import logging
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from model_lenet import RegressionModel
from svgd import MinNormSolver, get_gradient
import torch.autograd as autograd
from datetime import datetime


class H(nn.Module):
	def __init__(self, lr = 1e-3, input_dim=2):
		super().__init__()
		self.fc1_dim = input_dim
		self.fc2_dim = 20
		self.fc3_dim = 20
		self.fc4_dim = 1

		self.fc1 = nn.Linear(self.fc1_dim, self.fc2_dim)
		self.fc2 = nn.Linear(self.fc2_dim, self.fc3_dim)
		self.fc3 = nn.Linear(self.fc3_dim, self.fc4_dim)

		self.fc1.weight.data.normal_(0,0.1)
		self.fc2.weight.data.normal_(0,0.1)
		self.fc3.weight.data.normal_(0,0.1)

		self.lr = lr
		self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, betas=(0.0, 0.99))
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, X):
		X = self.fc1(X)
		X = F.relu(X)
		#X = self.fc2(X)
		#X = F.relu(X)
		return F.relu(self.fc3(X))+ 0.00001


def phi_network(X, H_nets, gradients, index):
	X = X.detach().requires_grad_(True)

	# adding noise to the original particles to get more training input
	Y1 = X + 0.1* torch.randn_like(X)
	Y2 = X + 0.1* torch.randn_like(X)
	Y3 = X + 0.1* torch.randn_like(X)
	Y4 = X + 0.1* torch.randn_like(X)
	Y5 = X + 0.1* torch.randn_like(X)
	Y = torch.vstack([Y1, Y2, Y3, Y4, Y5])
	Y = Y.detach().requires_grad_(True)

	for i in range(10):
		eps = torch.normal(0, 1, size=(Y.shape[0], Y.shape[1])) 
		log_H_eps = torch.log(H_nets[index](eps).mean())
		m_log_X = torch.log(H_nets[index](Y)).mean()
		loss = log_H_eps - m_log_X
		H_nets[index].optimizer.zero_grad()
		autograd.backward(loss)
		H_nets[index].optimizer.step()
	
	X.requires_grad = True
	output = torch.log(H_nets[index](X))
	grad = autograd.grad(outputs =output, inputs = X, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
	phi = -grad + 0.2*X + gradients[index] 
	return phi, H_nets[index]


def RBF(X, Y, bandwidth_scale, sigma=None):

	XX = X.matmul(X.t())
	YY = Y.matmul(Y.t())
	XY = X.matmul(Y.t())

	dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)
	if sigma is None:
		np_dnorm2 = dnorm2.detach().cpu().numpy()
		h = np.median(np_dnorm2)/(2* np.log(X.size(0) + 1))
		sigma = np.sqrt(h).item() * bandwidth_scale
	gamma = 1.0/(1e-8 + 2 *sigma **2)
	K_XY = (-gamma * dnorm2).exp()
	return K_XY


def test(weights):
	all_acc_1 = torch.zeros(args['num_nets'])
	all_acc_2 = torch.zeros(args['num_nets'])
	for i in range(args['num_nets']):
		net[i].eval()

	acc_1_ensemble = 0
	acc_2_ensemble = 0
	task1_ground_truth = []
	task2_ground_truth = []
	task1_preds = []
	task2_preds = []
	with torch.no_grad():
		for (it, batch) in enumerate(test_loader):

			X = batch[0]
			y = batch[1]

			out1_probs = []
			out2_probs = []
			for i in range(args['num_nets']):
				out1_prob, out2_prob = net[i](X)
				out1_prob = F.softmax(out1_prob, dim=1)
				out2_prob = F.softmax(out2_prob, dim=1)
				out1_probs.append(out1_prob * weights[i])
				out2_probs.append(out2_prob * weights[i])
				out1 = out1_prob.max(1)[1]
				out2 = out2_prob.max(1)[1]
				all_acc_1[i] += (out1 == y[:, 0]).sum()
				all_acc_2[i] += (out2 == y[:, 1]).sum()

			out1_prob = torch.stack(out1_probs).sum(0)
			out2_prob = torch.stack(out2_probs).sum(0)
			task1_ground_truth.append(y[:, 0].detach().clone())
			task2_ground_truth.append(y[:, 1].detach().clone())
			task1_preds.append(out1_prob.detach().clone())
			task2_preds.append(out2_prob.detach().clone())
			out1 = out1_prob.max(1)[1]
			out2 = out2_prob.max(1)[1]
			acc_1_ensemble += (out1 == y[:, 0]).sum()
			acc_2_ensemble += (out2 == y[:, 1]).sum()

		all_acc_1 = all_acc_1.cpu().numpy()[:, np.newaxis]
		all_acc_2 = all_acc_2.cpu().numpy()[:, np.newaxis]

		acc = np.concatenate((all_acc_1, all_acc_2), axis=1) / len(test_loader.dataset)
		acc_1_ensemble = acc_1_ensemble.item() / len(test_loader.dataset)
		acc_2_ensemble = acc_2_ensemble.item() / len(test_loader.dataset)
		task1_preds = torch.cat(task1_preds)
		task2_preds = torch.cat(task2_preds)
		task1_confidence, task1_preds = torch.max(task1_preds, 1)
		task2_confidence, task2_preds = torch.max(task2_preds, 1)

		task1_ground_truth = torch.cat(task1_ground_truth).cpu().numpy()
		task2_ground_truth = torch.cat(task2_ground_truth).cpu().numpy()

	return (acc_1_ensemble + acc_2_ensemble) / 2.0, acc_1_ensemble, acc_2_ensemble

def setup_seed(seed):
	torch.manual_seed(seed)
	#torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	#torch.backends.cudnn.deterministic = True



def train(epoch, weights, sol, method='lavengin', updateweight=True, H_nets=None):

	weights = torch.ones(args['num_nets'])/ args['num_nets']
	if epoch > 0:
		for i in range(args['num_nets']):
			shared_schedulers[i].step()
			classifier_schedulers[i].step()


	# training
	all_losses_1 = [0.0 for i in range(args['num_nets'])]
	all_losses_2 = [0.0 for i in range(args['num_nets'])]

	for (it, batch) in enumerate(test_loader):
		#if it >100:
		#	break
		X = batch[0]
		y = batch[1]


		batchsize_cur = X.shape[0]

		features =[]
		score_grads1_all =[]
		score_grads2_all =[]
		outputs1 =[]
		outputs2 =[]

		for i in range(args['num_nets']):
			net[i].train()
			net[i].zero_grad(set_to_none=False)
			features.append(net[i].encode(X))
			out1, out2 = net[i].decode(features[i].detach().clone())
			loss1 = criterion(out1, y[:, 0])
			all_losses_1[i] += loss1.detach().cpu().numpy() * batchsize_cur
			loss1.backward(retain_graph=True)

			loss2 = criterion(out2, y[:, 1])
			all_losses_2[i] += loss2.detach().cpu().numpy() * batchsize_cur
			loss2.backward(retain_graph=True)
			#print(epoch, it, i, loss1.item(), loss2.item())


			score_grads1 = None
			score_grads2 = None

			for name, param in net[i].named_parameters():
				if "task_1" in name:
					if score_grads1 is None:
						score_grads1 = param.grad.detach().data.clone().flatten()
					else:
						score_grads1 = torch.cat([score_grads1, param.grad.detach().data.clone().flatten()])
				if "task_2" in name:
					if score_grads2 is None:
						score_grads2 = param.grad.detach().data.clone().flatten()
					else:
						score_grads2 = torch.cat([score_grads2, param.grad.detach().data.clone().flatten()])
			net[i].zero_grad(set_to_none=False)
			outputs1.append(out1.reshape(-1))
			outputs2.append(out2.reshape(-1))
			score_grads1_all.append(score_grads1)
			score_grads2_all.append(score_grads2)

		score_grads1_all = torch.stack(score_grads1_all)
		score_grads2_all = torch.stack(score_grads2_all)

		outputs1 = torch.stack(outputs1, dim=0)
		outputs2 = torch.stack(outputs2, dim=0)

		w_matrix = weights[None, :]
		w_matrix.repeat(score_grads1_all.shape[0], 1)

		

		kernel1 = RBF(outputs1, outputs1.detach(), args['output_scale'])
		kernel2 = RBF(outputs2, outputs2.detach(), args['output_scale'])

		(kernel1 * w_matrix.detach()).sum().backward()
		(kernel2 * w_matrix.detach()).sum().backward()

		kernel_grads1_all =[]
		kernel_grads2_all =[]

		for i in range(args['num_nets']):
			kernel_grads1 = None
			kernel_grads2 = None

			for name, param in net[i].named_parameters():

				if "task_1" in name:
					if kernel_grads1 is None:
						kernel_grads1= param.grad.detach().data.clone().flatten()

					else:
						kernel_grads1 = torch.cat([kernel_grads1, param.grad.detach().data.clone().flatten()])
				if "task_2" in name:
					if kernel_grads2 is None:
						kernel_grads2= param.grad.detach().data.clone().flatten()
					else:
						kernel_grads2 = torch.cat([kernel_grads2, param.grad.detach().data.clone().flatten()])
			net[i].zero_grad(set_to_none=False)
			kernel_grads1_all.append(kernel_grads1)
			kernel_grads2_all.append(kernel_grads2)



		

		kernel_grads1_all = torch.stack(kernel_grads1_all)
		kernel_grads2_all = torch.stack(kernel_grads2_all)
		repeat_weights = weights[:, None].repeat(1, score_grads1_all.shape[1])

		if method == 'svgd':
			gradient1 = (kernel1.mm(score_grads1_all * repeat_weights) + args['output_tradeoff'] * kernel_grads1_all * repeat_weights)
			gradient2 = (kernel2.mm(score_grads2_all * repeat_weights) + args['output_tradeoff'] * kernel_grads2_all * repeat_weights)
		
		if method == 'blob':
			K_x_1 = 1.0/ kernel1.matmul(weights[:, None]) 
			hi1 = kernel_grads1_all * K_x_1.repeat(1, kernel_grads1_all.shape[1])
			gradient1 = score_grads1_all + args['output_tradeoff'] * hi1
			K_x_2 = 1.0/ kernel2.matmul(weights[:, None]) 
			hi2 = kernel_grads2_all * K_x_2.repeat(1, kernel_grads2_all.shape[1])
			gradient2 = score_grads2_all + args['output_tradeoff'] * hi2
		
		if method=='network':
			gradient1 = (kernel1.mm(score_grads1_all * repeat_weights) + args['output_tradeoff'] * kernel_grads1_all * repeat_weights)
			gradient2 = (kernel2.mm(score_grads2_all * repeat_weights) + args['output_tradeoff'] * kernel_grads2_all * repeat_weights)


		score_grads1_all = []
		score_grads2_all = []


		shared_loss_list_1 =torch.zeros(args['num_nets'])
		shared_loss_list_2 =torch.zeros(args['num_nets'])	


		for i in range(args['num_nets']):
			index1 = 0
			index2 = 0
			net[i].zero_grad(set_to_none=False)

			for name, param in net[i].named_parameters():
				if "task_1" in name:
	
					length = param.grad.flatten().shape[0]
					cur_grad = gradient1[i, index1:index1+length].view(param.grad.shape)
					param.grad.data = cur_grad.data.clone()
					index1 += length
				if "task_2" in name:
					length = param.grad.flatten().shape[0]
					cur_grad = gradient2[i, index2:index2+length].view(param.grad.shape)
					param.grad.data = cur_grad.data.clone()
					index2 += length
			classifier_optimizers[i].step()
			net[i].zero_grad(set_to_none=False)

			# encoder
			out1, out2 = net[i].decode(features[i])
			loss1 = criterion(out1, y[:, 0])
			loss1.backward(retain_graph=True)

			shared_loss_list_1[i] = loss1

			grads1 = None
			for name, param in net[i].named_parameters():
				if "task" not in name:
					if grads1 is None:
						grads1 = param.grad.detach().data.clone().flatten()
					else:
						grads1 = torch.cat([grads1, param.grad.detach().data.clone().flatten()])
				param.grad.zero_()


			loss2 = criterion(out2, y[:, 1])
			loss2.backward(retain_graph=True)

			shared_loss_list_2[i] = loss2


			grads2 = None
			for name, param in net[i].named_parameters():
				if "task" not in name:

					if grads2 is None:
						grads2 = param.grad.detach().data.clone().flatten()
					else:
						grads2 = torch.cat([grads2, param.grad.detach().data.clone().flatten()])
				param.grad.zero_()

			score_grads1_all.append(grads1)
			score_grads2_all.append(grads2)
			net[i].zero_grad(set_to_none=False)

		features = torch.stack([feature.flatten() for feature in features], dim=0)
		kernel = RBF(features, features.detach(), args['latent_scale'])
		wkernel = kernel * w_matrix.detach()
		wkernel.sum().backward()

		kernel_grads_all =[]
		params_all =[]

		for i in range(args['num_nets']):
			kernel_grads = None
			params = None
			for name, param in net[i].named_parameters():
				if "task" not in name:
					#print(i, name, param.shape, param.grad.shape)
					if params is None:
						params = param.detach().data.clone().flatten()
					else:
						params = torch.cat([params, param.detach().data.clone().flatten()])
					if kernel_grads is None:
						kernel_grads= param.grad.detach().data.clone().flatten()
					else:
						kernel_grads = torch.cat([kernel_grads, param.grad.detach().data.clone().flatten()])
				param.grad.zero_()
			
			kernel_grads_all.append(kernel_grads)
			params_all.append(params)
			net[i].zero_grad(set_to_none=False)
		

		kernel_grads_all = torch.stack(kernel_grads_all)
		params_all = torch.stack(params_all)

		# process the SVGD gradient
		score_grads1_all = torch.cat([score_grads1_all[i].unsqueeze(0) for i in range(args['num_nets'])], dim=0)
		score_grads2_all = torch.cat([score_grads2_all[i].unsqueeze(0) for i in range(args['num_nets'])], dim=0)

		if args['normalize']:
			score_grads1_all = torch.nn.functional.normalize(score_grads1_all, dim=0)
			score_grads2_all = torch.nn.functional.normalize(score_grads2_all, dim=0)

		if method == 'svgd':
			gradient1 = (wkernel.detach().mm(score_grads1_all) + args['latent_tradeoff'] * kernel_grads_all) #/args['num_nets']
			gradient2 = (wkernel.detach().mm(score_grads2_all) + args['latent_tradeoff'] * kernel_grads_all) #/args['num_nets']
		

		if method == 'blob':
			K_X = 1.0/kernel.matmul(weights[:,None])
			#K_X = K_X[:, None]
			kg = kernel_grads_all * K_X.repeat(1, kernel_grads_all.shape[1])
			gradient1 = score_grads1_all + args['latent_tradeoff'] * kg
			gradient2 = score_grads2_all + args['latent_tradeoff'] * kg

		if method == 'network':
			K_X = 1.0/kernel.matmul(weights[:,None])
			kg = kernel_grads_all * K_X.repeat(1, kernel_grads_all.shape[1])
			gradient1 = score_grads1_all + args['latent_tradeoff'] * kg
			gradient2 = score_grads2_all + args['latent_tradeoff'] * kg

			if H_nets is None:
				H_net1 = H(lr=1e-3, input_dim=params_all.shape[1])
				H_net2 = H(lr=1e-3, input_dim=params_all.shape[1])
				H_nets = [H_net1, H_net2]

			gradients = [gradient1, gradient2]
			gradient1, net1 = phi_network(params_all, H_nets, gradients, 0)
			gradient2, net2 = phi_network(params_all, H_nets, gradients, 1)
			H_nets = [net1, net2]

		Q = torch.zeros((2,2))
		shared_loss_list = torch.zeros(args['num_nets'])
		with torch.no_grad():
			if method=='svgd':
				kernel_m = wkernel
			if method=='blob':
				kernel_m = torch.ones(score_grads1_all.shape[0], score_grads1_all.shape[0])/(score_grads1_all.shape[0] * score_grads1_all.shape[0])
			if method=='network':
				kernel_m = torch.ones(score_grads1_all.shape[0], score_grads1_all.shape[0])

			Q[0][1] = torch.mul(kernel_m, torch.matmul(score_grads1_all, score_grads2_all.T)).sum()
			Q[1][0] = Q[0][1]
			Q[0][0] = torch.mul(kernel_m, torch.matmul(score_grads1_all, score_grads1_all.T)).sum()
			Q[1][1] = torch.mul(kernel_m, torch.matmul(score_grads2_all, score_grads2_all.T)).sum()


			
			logqz = torch.log(torch.sum(wkernel, dim=1))
			log_probs = torch.stack([-shared_loss_list_1, -shared_loss_list_2], 0)
			g_X = logqz[None, :].repeat(log_probs.shape[0], 1) - log_probs
			g_X_sol = torch.matmul(sol, g_X)
			exp_gx_sol = torch.exp(-g_X_sol)

			nom = torch.sum(exp_gx_sol[None, :].repeat(log_probs.shape[0], 1) * log_probs, dim=1)
			denom = torch.sum(exp_gx_sol)
			grad = nom/denom









			# find the optimal weight w for each iteration
			#sol, _ = MinNormSolver.find_min_norm_element(Q)
			#sol = MinNormSolver._projection2simplex(sol)
			
			# approximate the weight w by one gradient step, followed by the projection on the simplex
			tmp = torch.matmul(Q, sol)
			sol = sol - 0.00001*(tmp)
			sol = MinNormSolver._projection2simplex(sol)
			
			gradient = gradient1 * sol[0].item() + gradient2 * sol[1].item()
			#gradient = 0.5 * (gradient1 + gradient2)
		
		for i in range(args['num_nets']):
			index = 0
			net[i].zero_grad(set_to_none=False)
			for name, param in net[i].named_parameters():
				if "task" not in name:
					length = param.grad.flatten().shape[0]
					cur_grad = gradient[i, index : index + length].view(param.grad.shape)
					param.grad.data = cur_grad.data.clone()
					index += length
			shared_optimizers[i].step()
			net[i].zero_grad(set_to_none=False)

			shared_loss_list[i] = sol[0].item() * shared_loss_list_1[i] + sol[1].item() * shared_loss_list_2[i]

			if updateweight:
				weights[i] = weights[i] * torch.exp(-0.001 * (shared_loss_list[i] + logqz[i]))
		weights = weights/torch.sum(weights)
		#print("after", weights, torch.sum(weights))
		#exit(1)

	all_losses_1 = np.array(all_losses_1)[:, np.newaxis]
	all_losses_2 = np.array(all_losses_2)[:, np.newaxis]

	losses = np.concatenate((all_losses_1, all_losses_2), axis=1) / len(train_loader.dataset)
	return losses, weights, sol




				





args={}
args['lr'] = 0.001
args['batch_size'] = 256
args['num_nets'] = 5
args['n_epochs'] = 100


#select dataset for experiments

###-----------------------------------------
args['dset']= "multi_fashion_and_mnist"
#args['dset']="multi_fashion"
#args['dset']= "multi_mnist"
###-----------------------------------------

#select method for approximating the Wasserstein gradient: 'svgd' or 'blob' or 'network'
###-----------------------------------------
method='svgd'
#method='blob'
#method='network'
###-----------------------------------------


args['output_scale'] = 1.0
args['latent_scale'] = 1.0
args['output_tradeoff'] = 0.25
args['latent_tradeoff'] = 0.0015
args['normalize'] = True
H_nets = None # used for approximating the wasserstein gradient with neural networks

with open(f"./data/{args['dset']}.pickle", "rb") as f:
	trainX, trainLabel, testX, testLabel = pickle.load(f)

trainX = torch.from_numpy(trainX.reshape(120000, 1, 36, 36)).float()
trainLabel = torch.from_numpy(trainLabel).long()
testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
testLabel = torch.from_numpy(testLabel).long()
train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
test_set = torch.utils.data.TensorDataset(testX, testLabel)


test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size = args['batch_size'], shuffle=False)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size = args['batch_size'], shuffle=True)

criterion = nn.CrossEntropyLoss()
net = [RegressionModel(2) for _ in range(args['num_nets'])]

param_amount = 0
for p in net[0].named_parameters():
	param_amount += p[1].numel()
print("total amount of parameters: ", param_amount)

shared_optimizers = [torch.optim.SGD(net[i].get_shared_parameters(), lr = args['lr'], momentum=0.9) for i in range(args['num_nets'])]
classifier_optimizers = [torch.optim.SGD(net[i].get_classifier_parameters(), lr = args['lr'], momentum=0.9) for i in range(args['num_nets'])]
shared_schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 75, 90]) for optimizer in shared_optimizers]
classifier_schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 75, 90]) for optimizer in classifier_optimizers]
weights = torch.ones(args['num_nets'])/ args['num_nets']

print("----->  Dataset Name:", args['dset'])
print("----->  Method      :", method)
print("----->  No. Epoch   :", args['n_epochs'])

results = []
sol = torch.ones(2)/2

for i in range(args['n_epochs']):
	start=datetime.now()
	losses, weights, sol = train(i, weights , sol, method=method, updateweight=False, H_nets= H_nets)
	runningtime = datetime.now()-start

	avg, acc1, acc2 = test(weights)
	print("* Epoch ", str(i), ":", acc1, acc2, avg, sol)
	results.append([str(i), str(acc1), str(acc2), str(avg)])

# store information about the training process into the text file.
with open(args['dset']+"_"+ method +"_uniform_weight"+".txt", "w") as o:
	o.write(" ".join(["iter", "task 1 acc.", "task 2 acc.", "avg acc."]) + "\n")
	for line in results:
		o.write(" ".join(line) + "\n")
print("Finished!")


	






































