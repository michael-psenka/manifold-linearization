import torch
from torch import nn

# Python class that represents a layer of a neural network when merging
# multiple neighborhoods
# parameters:


# U: vector set of shape (D,k) representing normal directions for affine projector
# alpha: scalar set of shape (1,k) representing offset for affine projector
# p: number of neighboorhoods

# gamma: scalar multiple for softmax
# N_A: list of pytorch tensors representing shape of neighborhoods (inverse square
# root of covariance matrix)
# N_mu: list of pytorch tensors representing means of neighborhoods
# if above parameters not specified, no need to update membership.
class CCLayer(nn.Module):
	def __init__(self, U, alpha):
		super(CCLayer, self).__init__()
		self.U = U
		self.alpha = alpha
		self.D, self.p = U.shape

		
	def forward(self, ZPi):
		# ZPi: tensor of shape (D+p) x N, a concatenation of the data and
		# their neighborhood membership probabilities
		# we pass both through since we don't need to recompute the membership
		# at every layer

		# Z: R^(D x N)
		# Pi: R^(p x N)

		# returns: tensor of shape (D + p) x N

		Z = ZPi[:self.D, :]
		Pi = ZPi[self.D:, :]


		# if only one neighborhood, simply global projection operator
		if self.p == 1:
			# note we don't need offset for global linear operator
			return Z - self.U@self.U.T@Z

		# otherwise, we need smooth mixing of local operators
		N = Z.shape[1]

		# vector offsets for affine projectors
		b = self.U*self.alpha.reshape((1,self.p))
		b = b.reshape((self.D,1,self.p))
		
		# reshaped Z for broadcasting operations
		Z_reshape = Z.reshape((self.D, N, 1))
		
		# Compute the affine projection

		# (output is of shape (N, p))
		uTZ = (self.U.reshape((self.D, 1, self.p)) \
			* Z_reshape).sum(dim=0)
		# (output is of shape (D, N, p))
		uuTZ = self.U.reshape((self.D, 1, self.p)) \
			* uTZ.reshape((1, N, self.p))

		affine_proj = Z_reshape - uuTZ + b
		# Compute the output
		# (output is of shape (D, N))
		Z_out = (affine_proj * Pi.reshape((1, N, self.p))).sum(dim=2)
		return torch.vstack((Z_out, Pi))

# update membership estimation; note we don't need to update at every layer,
# since application of layer shouldn't change membership much of test points


# note that ZPi could either be just Z or ZPi
class CCUpdatePi(nn.Module):
	def __init__(self, gamma, N_A, N_mu):
		super(CCUpdatePi, self).__init__()
		self.gamma = gamma
		self.N_A = N_A
		self.N_mu = N_mu
		# extract dimensions
		self.D = N_mu[0].shape[0]
		self.p = len(N_A)

		
	def forward(self, ZPi):
		# reshaped Z for broadcasting operations
		Z = ZPi[:self.D, :]
		D, N = Z.shape

		N_eval_norms = torch.zeros((N, self.p))
		for i in range(self.p):
			N_eval_norms[:, i] = (self.N_A[i] @ (Z - self.N_mu[i])).pow(2).sum(dim=0)

		# vectorized
		# Z_reshape = Z.reshape((D, N, 1))
		# N_mu_reshape = self.N_mu.reshape((D, 1, self.p))
		# # evaluate neighborhood membership operators
		# N_eval_reshape = self.N_A.reshape((D, D, 1, self.p))*(Z.reshape((1, D, N, 1)) - N_mu_reshape)
		# # output is of shape (D x N x p)
		# N_eval = N_eval_reshape.sum(dim=1)

		# output is (N, p)
		# N_eval_norms = N_eval.pow(2).sum(axis=0)
		
		# need transpose to return to (p, N) standard
		Pi = torch.softmax(-self.gamma * N_eval_norms, dim=1).T

		# machine precision thresholding
		Pi[Pi < 1e-8] = 0
		Pi = Pi / Pi.sum(dim=0, keepdim=True)
		
		# Compute the output
		# (output is of shape (D + k, N))
		
		return torch.vstack((Z, Pi))

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

	def add_operation(self, nn_module, name = '_'):
		# if no name, just indicate which layer it is
		if name == '_':
			name = f'layer {len(self.network)}'
		self.network.add_module(name, nn_module)


# linear layer as in PyTorch, but with matmul convention flipped. here:
# y = A@x, instead of y = x@A^T
class LinearCol(nn.Module):
	def __init__(self, A):
		super(LinearCol, self).__init__()
		self.A = nn.Parameter(A)
		
	def forward(self, X):
		return self.A@X

# linear layer that projects out a subspace: x --> x - u@u^T@
class LinearProj(nn.Module):
	def __init__(self, u, D):
		super(LinearProj, self).__init__()
		self.u = nn.Parameter(u)
		# needed to extract Z from ZPi
		self.D = D
		
	def forward(self, ZPi):
		Z = ZPi[:self.D, :]
		return Z - self.u@self.u.T@Z

# translate and rescale data for downstream training
# akin to layer/batch normalization
class CCNormalize(nn.Module):
	def __init__(self, mu, scale):
		super(CCNormalize, self).__init__()
		self.mu = nn.Parameter(mu)
		self.scale = scale
		
	def forward(self, Z):
		return self.scale*(Z - self.mu)