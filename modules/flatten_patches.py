import torch
import torch.nn as nn
import torch.optim as optim

# gives initial flattening from points in neighborhoods
# input:
# X: data matrix of shape (D,n)
# ind_X: set of index sets corresponding to neighborhoods
# G_N0: adjacency matrix of neighborhoods

def flatten_from_points(X, ind_X, G_N0):
	# if data on gpu, default tensors to gpu too
	if X.is_cuda:
		torch.set_default_tensor_type('torch.cuda.FloatTensor')

	# get number of neighborhoods
	p = len(ind_X)
	# initialize normal directions
	U_0 = torch.randn(X.shape[0], p)
	# normalize columns
	U_0 = U_0/U_0.norm(dim=0, keepdim=True)

	flatten = FlattenFromPoints(X, ind_X, G_N0, U_0)
	opt = optim.SGD(flatten.parameters(), lr=0.001)

	for i in range(1000):
		flatten.zero_grad()
		# forward call of LinFlow
		loss = flatten()

		loss.backward()

		# GD step
		opt.step()
		# renormalization step
		with torch.no_grad():
			flatten.U.data = flatten.U.data/flatten.U.data.norm(dim=0, keepdim=True)

	# return flattening
	return flatten.U.data

def flatten_from_normals(X, N_index):
	return -1

# pytorch modules for optimizationm

# PyTorch model to optimize our custom loss

# U_0 is a matrix of shape (D, p), where p is the number
# of neighborhoods
class FlattenFromPoints(nn.Module):

	def __init__(self, X, ind_X, G_N0, U_0):
		super(FlattenFromPoints, self).__init__();
		# full data
		self.X = X
		self.ind_X = ind_X
		self.G_N0 = G_N0
		self.p = len(ind_X)
		# construct neighborhood data
		X_ = []
		edm_inv_ = []
		for i in range(self.p):
			# neighborhood data
			X_i = X[:,ind_X[i]]
			X_.append(X_i)
			# element-wise inverse of edm, used 
			edm_i = torch.cdist(X_i, X_i, p=2)
			n_i = edm_i.shape[0]
			edm_inv_.append(torch.divide(torch.Tensor([1]),torch.eye(n_i) + edm_i))

		self.X_ = X_
		self.edm_inv_ = edm_inv_
		
		# projection vec
		self.U = nn.Parameter(U_0)

	def forward(self):
		loss = 0

		# 1. compute the injectivity loss within each neighborhood
		for i in range(self.p):
			# stack single column of U to compare with dist vecs of X_i
			U_stacked = torch.ones(self.X_[i].shape)*self.U[:,[i]]
			# correlations betwee difference vectors of X_i and u
			corr_dXi_u = (self.X_[i].T @ U_stacked - U_stacked.T @ self.X_[i])*self.edm_inv_[i]
			# add to loss
			loss += 0.25*corr_dXi_u.pow(4).mean()

		# 2. compute the curvature loss between normal vectors
		U_gram_neighbors = (self.U.T @ self.U)[self.G_N0]
		loss += 0.5*U_gram_neighbors.pow(2).mean()

		return loss