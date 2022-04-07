# *****************************************************************************
#  CURVATURE COMPRESSION
# *****************************************************************************
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import find_patches, flatten_patches
from modules import cc_nn
# ****************************i*************************************************
# This is the primary script for the curvature compression algorithm.
# Input: data matrix X of shape (D,n), where D is the embedding dimension and
# n is the number of data points; OPTIONAL: N of shape (D,k) are k center
# points determining neighborhoods, and E of shape (k,k) a binary matrix
# denoting which neighborhoods are connected.

# Output: a neural network f: R^D -> R^d, where d is the intrinsic dimension of
# the data manifold X is drawn from.

# NOTE: this will always two a two-step flattening:
	# 1. flatten all patches, keeping curvature low
	# 2. find global flattening
# later, we will want to iteratively merge patches

def cc(X, N=-1, E=-1):
	# STEP 1: if not specified, determine appropriate neighborhoods for
	# local linearization
	# see modules/find_patves.py for details
	if N == -1:
		N, E = find_patches.knn(X)

	# STEP 2: compute the local linearization within each neighborhood
	U, alpha = flatten_patches.local_linearization(X, N, E)
	# STEP 3: compute the global flattening
	u_flat = flatten_patches.align_patches(U)
	# STEP 4: construct the neural network
	f = cc_nn.CC_2Network(U, alpha, u_flat)
	
	return f