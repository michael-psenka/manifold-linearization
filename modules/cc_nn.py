import torch
import torch.nn as nn
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

class FLayer(nn.Module):
	def __init__(self, U, x_mu, gamma, alpha=1):
		super(FLayer, self).__init__()
		self.U = U
		self.D, self.k = U.shape
		self.x_mu = x_mu
		self.gamma = gamma
		self.alpha = alpha

		
	def forward(self, X):
		# X: tensor of shape N x D, a batch of N points in D dimensions
		# that we want to flatten by 1 step

		# returns: tensor of shape N x D

		kernel = self.alpha*torch.exp(-self.gamma*(X - self.x_mu).pow(2).sum(dim=1, keepdim=True))
		# self.kernel = kernel
		proj = (X - self.x_mu)@self.U@self.U.T + self.x_mu
		Z = proj*kernel + X*(1 - kernel)

		return Z


class GLayer(nn.Module):
	def __init__(self, U, V, x_mu, x_c, gamma, alpha=1):
		super(GLayer, self).__init__()
		self.U = U
		self.V = V
		self.D, self.k = U.shape #U, V have same shape
		self.x_mu = x_mu
		self.x_c = x_c
		self.gamma = gamma
		self.alpha = alpha

		# computations we don't need to repeat every evaluation
		self.x_muU = x_mu @ U
		self.x_cU = x_c @ U
		self.change = x_c - x_mu - (self.x_cU - self.x_muU)@U.T

		# TESTING VAR: to use cross terms of second fundamental form
		# NOTE: ALSO NEED TO CHANGE IN cc.py IF CHANGING
		self.use_cross_terms = True

		
	def forward(self, Z):
		# Z: tensor of shape N x D, a batch of N points in D dimensions
		# that we want to flatten by 1 step

		# returns: tensor of shape N x D
		N = Z.shape[0]

		ZU = Z@self.U
		Z_norm2 = (Z-self.x_mu).pow(2).sum(dim=1, keepdim=True)
		ZU_norm2 = (ZU - self.x_muU).pow(2).sum(dim=1, keepdim=True)
		kernel = kernel_inv(Z_norm2, ZU_norm2, self.gamma, self.alpha)
		# self.kernel = kernel


		coord = ZU - self.x_cU
		if self.use_cross_terms:
			H_input = torch.bmm(coord.reshape((N,self.k,1)), coord.reshape((N,1,self.k)))
			idx_triu = torch.triu_indices(self.k,self.k)
			H_input = H_input[:,idx_triu[0,:],idx_triu[1,:]]

		else:
			H_input = coord.pow(2)
		
		Xhat = Z + kernel*(self.change + (H_input)@(self.V).T)

		return Xhat
# implementation of secant method. converge once step size of all
# below machine precision
# TODO: look into if this inversion is well-conditioned
def kernel_inv(Z_norm2, ZU_norm2, gamma, alpha):
	num_iter = 100
	# parameter setup
	ZUperp_norm2 = Z_norm2 - ZU_norm2
	exp_gammaz = torch.exp(-gamma*ZU_norm2)

	# initial guess
	guess_ratio = 1 - 1e-3
	# x_{n-2}
	x_m2 = ZUperp_norm2 * guess_ratio
	# x_{n-1}
	x_m1 = ZUperp_norm2

	x = x_m1
	# track size of each step
	step_size = 1

	# find inverse by root-finding using secant method
	
	for _ in range(num_iter):
		f_m1 = (1-alpha*exp_gammaz*torch.exp(-gamma*x_m1)).pow(2)*x_m1 - ZUperp_norm2
		f_m2 = (1-alpha*exp_gammaz*torch.exp(-gamma*x_m2)).pow(2)*x_m2 - ZUperp_norm2

		f_diff = f_m1 - f_m2
		f_diff[torch.logical_and(f_diff.abs() < 1e-6, f_diff >= 0)] = 1e-6
		f_diff[torch.logical_and(f_diff.abs() < 1e-6, f_diff < 0)] = -1e-6

		x = x_m1 - f_m1*(x_m1 - x_m2)/f_diff

		step_size = torch.abs(x - x_m1).max()
		x_m2 = x_m1
		x_m1 = x

		if step_size < 1e-6:
			break

	# return inverse
	return alpha*torch.exp(-gamma*(ZU_norm2 + x))