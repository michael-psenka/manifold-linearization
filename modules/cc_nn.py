# Python class that represents a layer of a neural network
# parameters:
# Np: (neighborhood points) point set of shape (D,k) representing k points in R^D of
# center of neighborhoods
# A: vector set of shape (D,k) representing normal directions for affine projector
# alpha: scalar set of shape (1,k) representing offset for affine projector
# gamma: scalar multiple for softmax

import torch
class CCLayer:
	def __init__(self, Np, A, alpha, gamma):
		self.Np = Np
		self.A = A
		self.alpha = alpha
		self.gamma = gamma
		self.D = Np.shape[0]
		self.k = Np.shape[1]
		
	def forward(self, x):
		# x: R^(D x N)
		# returns: R^(D x N)
		N = x.shape[1]

		# vector offsets for affine projectors
		b = self.A*self.alpha.reshape((1,self.k))
		b = b.reshape((self.D,1,self.k))
		
		# Compute the affine projection

		# (output is of shape (N, k))
		uTx = (self.A.reshape((self.D, 1, self.k)) \
			* x.reshape((self.D, N, 1))).sum(dim=0)
		# (output is of shape (D, N, k))
		uuTx = self.A.reshape((self.D, 1, self.k)) \
			* uTx.reshape((1, N, self.k))

		affine_proj = x.reshape((self.D, self.N, 1)) - uuTx + b
		
		# Compute the softmax
		# (output is of shape (N, k))
		softmax = torch.softmax(self.gamma * (affine_proj + self.b), dim=1)
		
		# Compute the output
		# (output is of shape (D, N))
		return (self.N.reshape((self.D, 1, self.k)) * softmax.reshape((1, self.k, 1))).sum(dim=2)