

import torch
from torch import nn

# Python class that represents a layer of a neural network when merging
# multiple neighborhoods
# parameters:
# Np: (neighborhood points) point set of shape (D,k) representing k points in R^D of
# center of neighborhoods
# A: vector set of shape (D,k) representing normal directions for affine projector
# alpha: scalar set of shape (1,k) representing offset for affine projector
# gamma: scalar multiple for softmax
class CCLayer(nn.Module):
	def __init__(self, Np, U, alpha, gamma):
		super(CCLayer, self).__init__()
		self.Np = Np
		self.U = U
		self.alpha = alpha
		self.gamma = gamma
		self.D = Np.shape[0]
		self.k = Np.shape[1]
		
	def forward(self, X):
		# X: R^(D x N)
		# returns: R^(D x N)


		# if only one neighborhood, simply global projection operator
		if self.k == 1:
			return 
		N = X.shape[1]

		# vector offsets for affine projectors
		b = self.U*self.alpha.reshape((1,self.k))
		b = b.reshape((self.D,1,self.k))
		
		# reshaped X for broadcasting operations
		X_reshape = X.reshape((self.D, N, 1))
		
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
		Np_reshape = self.Np.reshape((self.D, 1, self.k))
		softmax = torch.softmax(self.gamma * (Np_reshape - X_reshape).pow(2).sum(axis=0), dim=1)
		
		# Compute the output
		# (output is of shape (D, N))
		return (affine_proj * softmax.reshape((1, N, self.k))).sum(dim=2)

# we will iteratively build this network through calls to Sequential.add_module:
# e.g. ccn = CCNetwork(); ccn.network.add_module("layer1", CCLayer(Np, U, alpha, gamma))

# mu is a tensor of shape (D, 1) and corresponds to the mean of the data
# we need this since CCNetwork is constructed from the centered data, so given a new
# data point, we need to offset by the measured mean for our network to agree on the
# training data
class CCNetwork(nn.Module):
	def __init__(self, mu):
		super(CCNetwork, self).__init__()
		self.network = torch.nn.Sequential()
		self.mu = mu
		
	def forward(self, X):
		X_center = X - self.mu
		return self.network(X_center)

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