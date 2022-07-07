import torch
import torch.nn as nn
import torch.optim as optim

# gives initial flattening from points in neighborhoods
# input:
# Z: data matrix of shape (D,n)
# ind_Z: set of index sets corresponding to neighborhoods
# G_N0: adjacency matrix of neighborhoods
# U_global: collection of current global normal directions

def flatten_from_points(Z, ind_Z, G_N0, U_global):
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
	opt = optim.SGD(flatten.parameters(), lr=0.01)

	for i in range(3000):
		flatten.zero_grad()
		# forward call of LinFlow
		loss = flatten()

		loss.backward()

		# compute riemannian gradient
		# if i % 300 == 0:
		# 	print(f'loss: {loss}')
		# 	egrad = flatten.U.grad.detach()
		# 	base = flatten.U.data.detach()
		# 	uuTegrad = base * (base*egrad).sum(dim=0, keepdim=True)
		# 	rgrad = egrad - uuTegrad
		# 	# normalize based on num data points
		# 	rgrad_norm = rgrad.pow(2).mean(axis=0).sum().sqrt()
		# 	print(f'rgrad: {rgrad_norm}')

		if i % 100 == 0:
			print(f'loss: {loss}')

		# GD step
		opt.step()
		# renormalization step
		with torch.no_grad():
			# project out global normal directions
			if U_global.numel() > 1:
				flatten.U.data = flatten.U.data - U_global@U_global.T@flatten.U.data
			# normalize columns
			flatten.U.data = flatten.U.data/flatten.U.data.norm(dim=0, keepdim=True)
	print(f'U gram: {flatten.U.data.T @ flatten.U.data}')
	# return flattening
	return flatten.U.data

def flatten_from_normals(U_base, merge, G, U_global):
	# get number of neighborhoods
	p = len(merge)
	# initialize normal directions
	U_0 = torch.randn(U_base.shape[0], p)
	# normalize columns
	U_0 = U_0/U_0.norm(dim=0, keepdim=True)

	# construct pytorch optimization object
	flatten = FlattenFromNormals(U_base, merge, G, U_0)
	opt = optim.SGD(flatten.parameters(), lr=0.001)

	for i in range(1000):
		flatten.zero_grad()
		# forward call of LinFlow
		loss = flatten()

		loss.backward()

		# compute riemannian gradient
		# if i % 300 == 0:
		# 	print(f'loss: {loss}')
		# 	egrad = flatten.U.grad.detach()
		# 	base = flatten.U.data.detach()
		# 	uuTegrad = base * (base*egrad).sum(dim=0, keepdim=True)
		# 	rgrad = egrad - uuTegrad
		# 	# normalize based on num data points
		# 	rgrad_norm = rgrad.pow(2).mean(axis=0).sum().sqrt()
		# 	print(f'rgrad: {rgrad_norm}')

		if i % 100 == 0:
			print(f'loss: {loss}')

		# GD step
		opt.step()
		# renormalization step
		with torch.no_grad():
			# project out global normal directions
			if U_global.numel() > 1:
				flatten.U.data = flatten.U.data - U_global@U_global.T@flatten.U.data
			# normalize columns
			flatten.U.data = flatten.U.data/flatten.U.data.norm(dim=0, keepdim=True)

	# return flattening
	return flatten.U.data

def align_offsets(ZPi, U):
	p = U.shape[1]
	D = ZPi.shape[0] - p

	Z = ZPi[:D,:]
	Pi = ZPi[D:,:]
	# initialize normal directions
	alpha_init = torch.randn(p)

	########## 1 : compute offset for first neighborhood alpha_0

	# construct pytorch optimization object
	print(f'Pi shape: {Pi[[0],:].shape}')
	align_0 = AlignFirstOffset(Z, Pi[[0],:], U[:,[0]], alpha_init[0])
	opt_0 = optim.Adam(align_0.parameters(), lr=0.1)

	for i in range(1000):
		align_0.zero_grad()
		# forward call of LinFlow
		loss_0 = align_0()

		if i % 100 == 0:
			print(f'alpha_0: {align_0.alpha_0.data}')
			print(f'init align loss: {loss_0}')

		loss_0.backward()

		# # compute riemannian gradient
		# if i % 300 == 0:
		# 	print(f'loss: {loss}')
		# 	egrad = align.alpha.grad.detach()
		# 	rgrad = egrad
		# 	# normalize based on num data points
		# 	rgrad_norm = rgrad.pow(2).mean(axis=0).sum().sqrt()
		# 	print(f'rgrad: {rgrad_norm}')

		# GD step
		opt_0.step()


	alpha_0 = align_0.alpha_0.data.detach()
	#### 2: once first one fixed, align the rest

	# construct pytorch optimization object
	align = AlignOffsets(Z, Pi, U, alpha_init[1:], alpha_0)
	opt = optim.Adam(align.parameters(), lr=1)

	for i in range(1000):
		align.zero_grad()
		# forward call of LinFlow
		loss = align()

		loss.backward()

		# # compute riemannian gradient
		# if i % 300 == 0:
		# 	print(f'loss: {loss}')
		# 	egrad = align.alpha.grad.detach()
		# 	rgrad = egrad
		# 	# normalize based on num data points
		# 	rgrad_norm = rgrad.pow(2).mean(axis=0).sum().sqrt()
		# 	print(f'rgrad: {rgrad_norm}')

		# GD step
		opt.step()

		if i % 100 == 0:
			print(f'alpha: {align.alpha.data}')
			print(f'var loss: {loss}')

	# return alignment
	alpha_full = torch.zeros(p)
	alpha_full[0] = alpha_0
	alpha_full[1:] = align.alpha.data
	return alpha_full
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
		# note U represents a subspace, so U is equivalent to -U. 
		loss += 0.25*((1-U_gram_neighbors)/2).pow(4).mean()

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

		# boolean determining if this is a global projection or not
		self.global_proj = not (G.numel() > 1)

	def forward(self):
		loss = 0

		# 1. compute the injectivity loss within each neighborhood
		# 	note here we just want our normals U to be aligned with what we've chosen at the base
		for i in range(self.p):
			# sum of squares of inner products
			# loss -= 0.5*(self.U_base[:,self.merge[i]]*self.U[:,[i]]).sum(dim=0).pow(2).sum()
			# abs val of inner products with normals in merged neighborhood
			U_inner = (self.U_base[:,self.merge[i]]*self.U[:,[i]]).sum(dim=0).abs()
			# want them as close to 1 as possible.
			loss -= 0.25*((U_inner + 1) / 2).mean()

		# 2. compute the curvature loss between normal vectors
		if not self.global_proj:
			U_gram_neighbors = (self.U.T @ self.U)[self.G]
			# loss -= 0.5*U_gram_neighbors.pow(2).mean()
			loss -= 0.25*((U_gram_neighbors + 1)/2).mean()

		return loss

# alpha of shape (1)
class AlignFirstOffset(nn.Module):

	def __init__(self, Z, Pi_0, U_0, alpha_0):
		super(AlignFirstOffset, self).__init__();
		# full data. standardize shape to (D, N, p)
		# needed for downstream broadcasting
		self.D, self.N = Z.shape

		self.Z = Z
		# shape (1,N)
		self.Pi_0 = Pi_0
		# used to scale loss function with size of neighborhood
		self.Pi_0_sum_2 = Pi_0.sum().pow(2)
		# shape (D,1)
		self.U_0 = U_0
		# shape (1)
		self.alpha_0 = nn.Parameter(alpha_0)

		# get evaluation with proposed offset

		uuTZ = U_0 @ (U_0.T @ Z)

		self.Z_proj = self.Z - uuTZ

	def forward(self):
		
		Z_aff_proj = self.Z_proj + self.U_0*self.alpha_0

		# alpha loss function will be dependent on each other, trying to find
		# alphas that agree at intersections of the neighborhoods. We choose to
		# fix the first alpha[0] to also minimize the change to the original data
		# in the respective neighborhood

		loss = (1/self.Pi_0_sum_2)*((self.Z - Z_aff_proj).pow(2)*self.Pi_0).sum()
		
		return loss

# alpha of shape (p)
class AlignOffsets(nn.Module):

	def __init__(self, Z, Pi, U, alpha_init, alpha_0):
		super(AlignOffsets, self).__init__();
		# full data. standardize shape to (D, N, p)
		# needed for downstream broadcasting
		self.D, self.N = Z.shape
		self.p = Pi.shape[0]

		self.Z = Z.reshape((self.D, self.N, 1))
		self.Pi = (Pi.T).reshape((1, self.N, self.p))
		self.U = U.reshape((self.D, 1, self.p))

		self.alpha = nn.Parameter(alpha_init.reshape((1,1,self.p-1)))
		# init offset for first neighborhood
		self.alpha_0 = alpha_0.reshape((1,1,1))

		# get evaluation with proposed offset
		# output of shape (1, N, p)
		uTZ = (self.Z * self.U).sum(dim=0, keepdim=True)
		# output of shape (D, N, p)
		uuTZ = self.U * uTZ

		self.Z_proj = self.Z - uuTZ

	def forward(self):
		
		Z_aff_proj = self.Z_proj + \
			torch.cat((self.U[:,:,[0]]*self.alpha_0,self.U[:,:,1:]*self.alpha), dim=2)
		# once we choose global position of 1st, we can fix as global frame
		# and only optimize the rest for the "variance" loss
		
		# now we minimize the "variance" from different projectors at each point, w.r.t
		# the distribution from the softmax (Pi)

		# output of shape (D, N, 1)
		E_Zproj = (Z_aff_proj*self.Pi).sum(dim=2, keepdim=True)
		Var_Z = (1/self.p)*((Z_aff_proj - E_Zproj).pow(2)*self.Pi).mean(dim=1).sum()
		loss = 0.5*Var_Z
		
		return loss