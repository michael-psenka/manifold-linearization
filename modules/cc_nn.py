

import torch
from torch import nn

# Python class that represents a layer of a neural network when merging
# multiple neighborhoods
# parameters:


# U: vector set of shape (D,k) representing normal directions for affine projector
# alpha: scalar set of shape (1,k) representing offset for affine projector
# k: number of neighboorhoods

# gamma: scalar multiple for softmax
# N_A: list of pytorch tensors representing shape of neighborhoods (inverse square
# root of covariance matrix)
# N_mu: list of pytorch tensors representing means of neighborhoods
# if above parameters not specified, no need to update membership.
class CCLayer(nn.Module):
	def __init__(self, U, alpha, k, gamma=-1, N_A=-1, N_mu=-1):
		super(CCLayer, self).__init__()
		self.U = U
		self.alpha = alpha
		self.D = U.shape[0]
		self.k = k
		self.gamma = gamma
		self.N_A = N_A
		self.N_mu = N_mu

		
	def forward(self, XPi):
		# XPi: tensor of shape (D+k) x N, a concatenation of the data and
		# their neighborhood membership probabilities
		# we pass both through since we don't need to recompute the membership
		# at every layer

		# X: R^(D x N)
		# Pi: R^(k x N)

		# returns: tensor of shape (D + k) x N

		X = XPi[:self.D, :]
		Pi = XPi[self.D:, :]


		# if only one neighborhood, simply global projection operator
		if self.k == 1:
			# note we don't need offset for global linear operator
			return X - self.U@self.U.T@X

		# otherwise, we need smooth mixing of local operators
		N = X.shape[1]

		# vector offsets for affine projectors
		b = self.U*self.alpha.reshape((1,self.k))
		b = b.reshape((self.D,1,self.k))
		
		# reshaped X for broadcasting operations
		X_reshape = X.reshape((self.D, N, 1))
		N_mu_reshape = self.N_mu.reshape((self.D, 1, self.k))
		# evaluate neighborhood membership operators
		N_eval_reshape = self.N_A.reshape((self.D, self.D, 1, k))*(X.reshape((1, self.D, N, 1)) - N_mu_reshape)
		# output is of shape (D x N x k)
		N_eval = N_eval_reshape.sum(dim=1)
		
		# Compute the affine projection

		# (output is of shape (N, k))
		uTX = (self.U.reshape((self.D, 1, self.k)) \
			* X_reshape).sum(dim=0)
		# (output is of shape (D, N, k))
		uuTX = self.U.reshape((self.D, 1, self.k)) \
			* uTX.reshape((1, N, self.k))

		affine_proj = X_reshape - uuTX + b

		# Computes probability of each point in batch being in each neighborhood
		# (output is of shape (N, k))
		

		# if we specify gamma, then we want to update Pi
		if not self.gamma == -1:
			# output is of shape (k x N)
			Pi = torch.softmax(self.gamma * (N_eval).pow(2).sum(axis=0), dim=1).T
		
		# Compute the output
		# (output is of shape (D, N))
		X_out = (affine_proj * Pi.reshape((1, N, self.k))).sum(dim=2)
		return torch.vstack((X_out, Pi))

# first call to neighborhood membership estimation, takes data matrix
class CCInitPi(nn.Module):
	def __init__(self, gamma, N_A, N_mu):
		super(CCInitPi, self).__init__()
		self.gamma = gamma
		self.N_A = N_A
		self.N_mu = N_mu

		
	def forward(self, X):
		# reshaped X for broadcasting operations
		X_reshape = X.reshape((self.D, N, 1))
		N_mu_reshape = self.N_mu.reshape((self.D, 1, self.k))
		# evaluate neighborhood membership operators
		N_eval_reshape = self.N_A.reshape((self.D, self.D, 1, k))*(X.reshape((1, self.D, N, 1)) - N_mu_reshape)
		# output is of shape (D x N x k)
		N_eval = N_eval_reshape.sum(dim=1)
		
		Pi = torch.softmax(self.gamma * (N_eval).pow(2).sum(axis=0), dim=1).T
		
		# Compute the output
		# (output is of shape (D + k, N))
		
		return torch.vstack((X, Pi))

# we will iteratively build this network through calls to Sequential.add_module:
# e.g. ccn = CCNetwork(); ccn.network.add_module("layer1", CCLayer(Np, U, alpha, gamma))

# mu is a tensor of shape (D, 1) and corresponds to the mean of the data
# we need this since CCNetwork is constructed from the centered data, so given a new
# data point, we need to offset by the measured mean for our network to agree on the
# training data
class CCNetwork(nn.Module):
	def __init__(self):
		super(CCNetwork, self).__init__()
		self.network = torch.nn.Sequential()
		
	def forward(self, X):
		return self.network(X)

	def add_operation(self, nn_module):
		self.network.add_module(f'layer {len(self.network)}', nn_module)


# linear layer as in PyTorch, but with matmul convention flipped. here:
# y = A@x, instead of y = x@A^T
class LinearCol(nn.Module):
	def __init__(self, A):
		super(LinearCol, self).__init__()
		self.A = nn.Parameter(A)
		
	def forward(self, x):
		return self.A@x

class CCRecenter(nn.Module):
	def __init__(self, mu):
		super(CCRecenter, self).__init__()
		self.mu = nn.Parameter(mu)
		
	def forward(self, x):
		return x - self.mu