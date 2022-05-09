import torch
import torch.nn as nn
import torch.optim as optim

# gives initial flattening from points in neighborhoods
# input:
# Z: data matrix of shape (D,n)
# ind_Z: set of index sets corresponding to neighborhoods
# G_N0: adjacency matrix of neighborhoods

def flatten_from_points(Z, ind_Z, G_N0):
	# if data on gpu, default tensors to gpu too
	if Z.is_cuda:
		torch.set_default_tensor_type('torch.cuda.FloatTensor')

	# get number of neighborhoods
	p = len(ind_Z)
	# initialize normal directions
	U_0 = torch.randn(Z.shape[0], p)
	# normalize columns
	U_0 = U_0/U_0.norm(dim=0, keepdim=True)

	# construct pytorch optimization object
	flatten = FlattenFromPoints(Z, ind_Z, G_N0, U_0)
	opt = optim.SGD(flatten.parameters(), lr=0.1)

	for i in range(1000):
		flatten.zero_grad()
		# forward call of LinFlow
		loss = flatten()

		loss.backward()

		# compute riemannian gradient
		if i % 300 == 0:
			print(f'loss: {loss}')
			egrad = flatten.U.grad.detach()
			base = flatten.U.data.detach()
			uuTegrad = base * (base*egrad).sum(dim=0, keepdim=True)
			rgrad = egrad - uuTegrad
			# normalize based on num data points
			rgrad_norm = rgrad.pow(2).mean(axis=0).sum().sqrt()
			print(f'rgrad: {rgrad_norm}')

		# GD step
		opt.step()
		# renormalization step
		with torch.no_grad():
			flatten.U.data = flatten.U.data/flatten.U.data.norm(dim=0, keepdim=True)

	# return flattening
	return flatten.U.data

def flatten_from_normals(U_base, merge, G):
	# get number of neighborhoods
	p = len(merge)
	# initialize normal directions
	U_0 = torch.randn(U_base.shape[0], p)
	# normalize columns
	U_0 = U_0/U_0.norm(dim=0, keepdim=True)

	# construct pytorch optimization object
	flatten = FlattenFromNormals(U_base, merge, G, U_0)
	opt = optim.SGD(flatten.parameters(), lr=0.1)

	for i in range(1000):
		flatten.zero_grad()
		# forward call of LinFlow
		loss = flatten()

		loss.backward()

		# compute riemannian gradient
		if i % 300 == 0:
			print(f'loss: {loss}')
			egrad = flatten.U.grad.detach()
			base = flatten.U.data.detach()
			uuTegrad = base * (base*egrad).sum(dim=0, keepdim=True)
			rgrad = egrad - uuTegrad
			# normalize based on num data points
			rgrad_norm = rgrad.pow(2).mean(axis=0).sum().sqrt()
			print(f'rgrad: {rgrad_norm}')

		# GD step
		opt.step()
		# renormalization step
		with torch.no_grad():
			flatten.U.data = flatten.U.data/flatten.U.data.norm(dim=0, keepdim=True)

	# return flattening
	return flatten.U.data

def align_offsets(ZPi, U):
	p = U.shape[1]
	D = ZPi.shape[0] - p

	Z = ZPi[:D,:]
	Pi = ZPi[D:,:]
	# initialize normal directions
	alpha_0 = torch.randn(p)

	# construct pytorch optimization object
	align = AlignOffsets(Z, Pi, U, alpha_0)
	opt = optim.SGD(align.parameters(), lr=0.1)

	for i in range(1000):
		align.zero_grad()
		# forward call of LinFlow
		loss = align()

		loss.backward()

		# compute riemannian gradient
		if i % 300 == 0:
			print(f'loss: {loss}')
			egrad = align.alpha.grad.detach()
			rgrad = egrad
			# normalize based on num data points
			rgrad_norm = rgrad.pow(2).mean(axis=0).sum().sqrt()
			print(f'rgrad: {rgrad_norm}')

		# GD step
		opt.step()

	# return alignment
	return align.alpha.data
# pytorch modules for optimizationm

# PyTorch model to optimize our custom loss

# U_0 is a matrix of shape (D, p), where p is the number
# of neighborhoods
class FlattenFromPoints(nn.Module):

	def __init__(self, Z, ind_Z, G_N0, U_0):
		super(FlattenFromPoints, self).__init__();
		# full data
		self.Z = Z
		self.ind_Z = ind_Z
		self.G_N0 = G_N0
		self.p = len(ind_Z)
		# construct neighborhood data
		Z_ = []
		edm_inv_ = []
		for i in range(self.p):
			# neighborhood data
			Z_i = Z[:,ind_Z[i]]
			Z_.append(Z_i)
			# element-wise inverse of edm, used 
			edm_i = torch.cdist(Z_i.T, Z_i.T, p=2)
			n_i = edm_i.shape[0]
			edm_inv_.append(torch.divide(torch.Tensor([1]),torch.eye(n_i) + edm_i))

		self.Z_ = Z_
		self.edm_inv_ = edm_inv_
		
		# projection vec
		self.U = nn.Parameter(U_0)

	def forward(self):
		loss = 0

		# 1. compute the injectivity loss within each neighborhood
		for i in range(self.p):
			# stack single column of U to compare with dist vecs of Z_i
			U_stacked = torch.ones(self.Z_[i].shape)*self.U[:,[i]]
			# correlations betwee difference vectors of Z_i and u
			corr_dZi_u = (self.Z_[i].T @ U_stacked - U_stacked.T @ self.Z_[i])*self.edm_inv_[i]
			# add to loss
			loss += 0.25*corr_dZi_u.pow(4).mean()

		# 2. compute the curvature loss between normal vectors
		U_gram_neighbors = (self.U.T @ self.U)[self.G_N0]
		loss -= 0.5*U_gram_neighbors.pow(2).mean()

		return loss


class FlattenFromNormals(nn.Module):

	def __init__(self, U_base, merge, G, U_0):
		super(FlattenFromNormals, self).__init__();
		# full data
		self.U_base = U_base
		self.merge = merge
		self.G = G
		self.p = len(merge)
		
		# projection vec
		self.U = nn.Parameter(U_0)

	def forward(self):
		loss = 0

		# 1. compute the injectivity loss within each neighborhood
		# 	note here we just want our normals U to be aligned with what we've chosen at the base
		for i in range(self.p):
			# sum of squares of inner products
			loss -= 0.5*(self.U_base[:,self.merge[i]]*self.U[:,[i]]).sum(dim=0).pow(2).sum()

		# 2. compute the curvature loss between normal vectors
		U_gram_neighbors = (self.U.T @ self.U)[self.G]
		loss -= 0.5*U_gram_neighbors.pow(2).mean()

		return loss


# alpha of shape (p)
class AlignOffsets(nn.Module):

	def __init__(self, Z, Pi, U, alpha_0):
		super(AlignOffsets, self).__init__();
		# full data. standardize shape to (D, N, p)
		# needed for downstream broadcasting
		self.D, self.N = Z.shape
		self.p = Pi.shape[0]

		self.Z = Z.reshape((self.D, self.N, 1))
		self.Pi = Pi.T.reshape((1, self.N, self.p))
		self.U = U.reshape((self.D, 1, self.p))

		self.alpha = nn.Parameter(alpha_0.reshape((1,1,self.p)))

		# get evaluation with proposed offset
		# output of shape (1, N, p)
		uTZ = (self.Z * self.U).sum(dim=0, keepdim=True)
		# output of shape (D, N, p)
		uuTZ = self.U * uTZ

		self.Z_proj = self.Z - uuTZ

	def forward(self):
		
		Z_aff_proj = self.Z_proj + self.U*self.alpha

		# alpha loss function will be dependent on each other, trying to find
		# alphas that agree at intersections of the neighborhoods. We choose to
		# fix the first alpha[0] to also minimize the change to the original data
		# in the respective neighborhood
		loss = ((self.Z[:,:,0] - Z_aff_proj[:,:,0]).pow(2)*self.Pi[:,:,0]).mean(dim=1).sum()

		# once we choose global position of 1st, we can fix as global frame
		# and only optimize the rest for the "variance" loss
		
		# now we minimize the "variance" from different projectors at each point, w.r.t
		# the distribution from the softmax (Pi)

		# output of shape (D, N, 1)
		E_Zproj = (Z_aff_proj*self.Pi).sum(dim=2, keepdim=True)
		Var_Z = ((Z_aff_proj - E_Zproj).pow(2)*self.Pi).mean(dim=2).mean(dim=1).sum()

		loss += 0.5*Var_Z
		
		return loss