# *****************************************************************************
#  CURVATURE COMPRESSION
# *****************************************************************************
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from modules import cc_nn, pca_init, flatten_patches, find_patches_community_detection

from tqdm import trange
# ****************************i*************************************************
# This is the primary script for the curvature compression algorithm.
# Input: data matrix X of shape (n,D), where D is the embedding dimension and
# n is the number of data points;
# d_desired is the desired dimension to flatten X onto

# Output: a neural network f: R^D -> R^d, where d is the intrinsic dimension of
# the data manifold X is drawn from.

# gamma_0 is the starting value of the "inverse neighborhood size"
# --- (the smaller gamma_0 is, the larger the neighborhood size is)


def cc(X, d_target, gamma_0=1, n_iters=1000):
	#######	## HYPERPARAMTERS ####
	##############################

	# set up needed variables
	N, D = X.shape
	# TODO: reimplement CCnet
	cc_network = cc_nn.CCNetwork()

	# NOTE: for now, we will just track the training data
	Z = X.detach().clone()

	print('Starting optimization...')
	
	with trange(n_iters, unit="iters") as pbar:
		for _ in pbar:
			# center of neighborhood selection
			center = Z[torch.randint(N, (1,)), :]

			# weighted kernel from current selected point
			kernel_eval = torch.exp(-gamma_0*torch.sum((Z - center)
							** 2, dim=1, keepdim=True))
			# tensor of shape R^(n x 1)
			kernel_eval = kernel_eval / torch.sum(kernel_eval)

			# weighted mean from current selected point
			kernel_mean = torch.sum(kernel_eval*Z, dim=0, keepdim=True)

			# find new kernel from new center
			kernel_new = torch.exp(-gamma_0*torch.sum((Z - kernel_mean)
						** 2, dim=1, keepdim=True))
			kernel_new = kernel_new / torch.sum(kernel_new)

			# kernel pca
			U, S, Vt = torch.linalg.svd(torch.sqrt(kernel_new)*(Z - kernel_mean))

			pca_kernel = Vt.T[:, :d_target]
			# affine map
			subspace_proj = ((Z - kernel_mean)@pca_kernel)@pca_kernel.T + kernel_mean

			# finally, compute POU using kernel
			Z_new = (1-kernel_new)*Z + kernel_new*subspace_proj

			pbar.set_postfix({"step size": torch.norm(Z_new - Z).item() / np.sqrt(N*D)})
			Z = Z_new

	return Z
