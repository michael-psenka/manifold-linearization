# *****************************************************************************
#  CURVATURE COMPRESSION
# *****************************************************************************
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import cc_nn, find_patches, flatten_patches, \
	merge_patches, pca_init
# ****************************i*************************************************
# This is the primary script for the curvature compression algorithm.
# Input: data matrix X of shape (D,n), where D is the embedding dimension and
# n is the number of data points; OPTIONAL: Ne of shape (D,k) are k center
# points determining neighborhoods, and E of shape (k,k) a binary matrix
# denoting which neighborhoods are connected.

# Output: a neural network f: R^D -> R^d, where d is the intrinsic dimension of
# the data manifold X is drawn from.

# NOTE: this will always two a two-step flattening:
	# 1. flatten all patches, keeping curvature low
	# 2. find global flattening
# later, we will want to iteratively merge patches

def cc(X, Np=-1, E=-1):
	# HYPERPARAMTERs
	_gamma = 1
	# set up needed variables
	d, N = X.shape
	mu = X.mean(axis=1, keepdim=True)

	cc_network = cc_nn.CCNetwork(mu)
	# STEP 0: use PCA to project out machine-precision directions
	X, U_r = pca_init(X - mu)
	# add PCA operation to network
	cc_network.add_module("pca", cc_nn.LinearCol(U_r.T))
	# new ambient dimension, rank of PCA
	d_current = U.shape[1]


	# MAIN LOOP: looping through ambient dimension d_current until
	# it has been reduced as much as possible
	has_converged = False
	while not has_converged:
		# STEP 1: find patches
		Np, E = find_patches(X)

		# INNER LOOP: flatten and merge patches
		k = Np.shape[1]

		# our first flattening is from the points within each neighborhood
		# we also check here to see if the manifold has been flattened
		# everywhere -- if so, impossible to compress further
		U, alpha, has_converged = flatten_patches.flatten_from_points(X, Np, E)
		# if test for convergence is true, we don't want to add the above
		# generated layer
		if has_converged:
			break

		# add layer to network
		cc_layer = cc_nn.CCLayer(Np, U, alpha, _gamma)
		# add to network
		cc_network.add_operation(cc_layer)

		# if k > 1, linearize and merge neighborhoods until we merge into
		# one neighborhood
		if k > 1:
			while k > 1:
				# merge patches
				# M is a structure that contains information on what clusters
				# we merge patches into
				M, Np_new, E_new = merge_patches(Np, E)
				# flatten patches
				U_new, alpha = flatten_patches.flatten_from_normals(X, U, M, Np_new, E_new)
				# STEP 3: construct global map from local flattenings
				cc_layer = cc_nn.CCLayerMulti(Np_new, U_new, alpha, _gamma)
				# add to network
				cc_network.add_operation(cc_layer)
				
				# update parameters
				k = Np.shape[1]
				Np = Np_new
				E = E_new
				U = U_new

			flatten_patches.flatten_from_normals(X, N, E)
	
	return cc_network