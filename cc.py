# *****************************************************************************
#  CURVATURE COMPRESSION
# *****************************************************************************
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import cc_nn, pca_init, flatten_patches, find_patches_community_detection
# ****************************i*************************************************
# This is the primary script for the curvature compression algorithm.
# Input: data matrix X of shape (D,n), where D is the embedding dimension and
# n is the number of data points; OPTIONAL: Ne of shape (D,k) are k center
# points determining neighborhoods, and E of shape (k,k) a binary matrix
# denoting which neighborhoods are connected.

# Output: a neural network f: R^D -> R^d, where d is the intrinsic dimension of
# the data manifold X is drawn from.

# k is number of neighborhoods for kNN graph to approximate manifold

def cc(X, k):
	######### HYPERPARAMTERS ####
	# used for softmax
	_gamma = 2
	# noise tolerance fo rdata
	_eps = 1e-2
	#  how much to widen neighborhoods for neighboring detection
	_eps_N = 0
	##############################

	# set up needed variables
	D, N = X.shape
	cc_network = cc_nn.CCNetwork()

	mu = X.mean(axis=1, keepdim=True)
	# scaled global norm based on number of samples
	global_norm = (X - mu).norm(p=2) / N
	
	# add centering to network
	cc_network.add_operation(cc_nn.CCNormalize(mu, 1/global_norm))
	# now we start calling X as Z as we pass it through our constructed networkj
	Z = X - mu
	print('Init PCA to reduce nonnecessary dimensions...')
	# STEP 0: use PCA to project out machine-precision directions
	Z, U_r = pca_init.pca_init(Z)
	# add PCA operation to network
	cc_network.add_operation(cc_nn.LinearCol(U_r.T), "pca")
	# new ambient dimension, rank of PCA
	d_current = U_r.shape[1]

	# find neighborhood index sets
	print('Finding neighborhoods...')

	ind_Z, merges, A_N, mu_N, G_N = \
		find_patches_community_detection.find_neighborhood_structure(Z, k, _eps, _eps_N)

	# test membership layer
	print('Finding membership...')
	layer_pi = cc_nn.CCUpdatePi(_gamma, A_N, mu_N)
	print('Applying membership...')
	ZPi = layer_pi(Z)
	cc_network.add_operation(layer_pi, f"pi;d:{d_current}")

	# test flatten from points

	print('Beginning main construction loop.')
	# MAIN LOOP: looping through ambient dimension d_current until
	# it has been reduced as much as possible
	has_converged = False
	while not has_converged:

		# STEP 1: find init normal directions from neighborhood points
		# here we optimize for injectivity with the minimal induced extrinsic curvaturew
		print('Finding normals...')
		Z = ZPi[:d_current,:]
		U = flatten_patches.flatten_from_points(Z, ind_Z, G_N[0])

		print('Aligning projectors...')
		alpha = flatten_patches.align_offsets(ZPi, U)

		print('Flatten from normals...')
		U2 = flatten_patches.flatten_from_normals(U, merges[0], G_N[1])

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