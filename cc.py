# *****************************************************************************
#  CURVATURE COMPRESSION
# *****************************************************************************
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from modules import cc_nn

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


def cc(X):
	N, D = X.shape
	# needed for dist-to-gamma conversion
	log2 = float(np.log(2))
	#######	## HYPERPARAMTERS ####
	##############################

	# how many iterations to binary search for biggest possible radius
	fid = 10
	# number of flattening steps to perform (currently no stoppic criterion)
	n_iter = 50
	# how many max steps for inner optimization of U, V
	# (stopping criterion implemented)
	n_iter_inner = 1000
	# threshold for reconstruction loss being good enough
	thres_recon = 1e-4

	# global parameter for kernel
	alpha = 0.5


	############# INIT GLOBAL VARIABLES##########
	# encoder network
	f = cc_nn.CCNetwork()
	# decoder network
	g = cc_nn.CCNetwork()
	Z = X.clone()
	# ################ MAIN LOOP #########################
	with trange(n_iter, unit="iters") as pbar:
		for _ in pbar:

			# start our "neighborhood size" at max possible
			d_max = torch.cdist(Z, Z).max()
			d_init = d_max.clone() / 2
			d_curr = d_init.clone()

			# 1 step of flattening/regeneration
			choice = torch.randint(N, (1,))
			# choice=25
			z_c = Z[choice,:]

			U_0 = torch.randn(2, 1)
			U_0 = U_0 / torch.norm(U_0, p=2)
			U = torch.nn.Parameter(U_0.clone())
			U.requires_grad = True


			opt_U = optim.SGD([U], lr=1)
		# outer loop, searching for max radius with desired fidelity
			for i in range(fid):
				# set up kernel
				gamma = log2/d_curr**2
				kernel_pre = torch.exp(-gamma*(Z - z_c).pow(2).sum(dim=1, keepdim=True))
				z_mu = (Z*kernel_pre).sum(dim=0, keepdim=True) / kernel_pre.sum()
				kernel = alpha*torch.exp(-gamma*(Z - z_mu).pow(2).sum(dim=1, keepdim=True))

				U_old = U.data.clone()
				# inner loop, find best U, V to reconstruct
				for _ in range(n_iter_inner):
					opt_U.zero_grad()
					# opt_V.zero_grad()
					# opt_alpha.zero_grad()
					coord = (Z - z_c) @ U
					coord2 = coord.pow(2)
					Z_perp = (Z - z_c) - coord @ U.T
					A = coord2 * kernel
					b = Z_perp * kernel

					# least squares solution for V, note automatically orthogonal to U
					# with torch.no_grad():
					V = ((A.T@A).inverse() @ (A.T@b)).T

					loss = 0.5*(kernel*(Z_perp - coord2@V.T)).pow(2).mean()
					# loss = (U).pow(2).mean()
					loss.backward()

					opt_U.step()
					with torch.no_grad():
						U.data = U.data / torch.norm(U.data, p=2)
						step_size = (U.data - U_old).pow(2).mean().sqrt()
						U_old = U.data.clone()

						if step_size < 1e-5:
							break
				
				# achieved desired recon loss, so increase neighborhood
				# size
				if loss.item() <= thres_recon:
					d_curr += d_init * (0.5)**(i+1)
				# didn't achieve desired recon loss, so decrease neighborhood
				else:
					d_curr -= d_init * (0.5)**(i+1)

				# if radius already bigger than biggest possible, no need to continue searching
				if d_curr > d_max:
					break

			Z = Z.detach()
			U = U.detach()
			V = V.detach()

			f_layer = cc_nn.FLayer(U, z_mu, gamma, alpha)
			g_layer = cc_nn.GLayer(U, V, z_mu, z_c, gamma, alpha)

			f.add_operation(f_layer)
			g.add_operation(g_layer)
			# update data
			Z = f_layer(Z)
			pbar.set_postfix({"loss": loss.item(), "d_ratio": (d_curr/d_max).item()})

	return f, g