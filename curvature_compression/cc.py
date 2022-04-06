# *****************************************************************************
#  CURVATURE COMPRESSION
# *****************************************************************************
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import flatten-patches



# *****************************************************************************
# This is the primary script for the curvature compression algorithm.
# Input: data matrix X of shape (D,n), where D is the embedding dimension and
# n is the number of data points; 

# Output: a neural network f: R^D -> R^d, where d is the intrinsic dimension of
# the data manifold X is drawn from.
def cc(X):
	# STEP 1: determine appropriate neighborhoods
	# N, E = 