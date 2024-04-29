# *****************************************************************************
#  CURVATURE COMPRESSION
# *****************************************************************************


# Important lines:
# *---------------------------------------------------------*

# 52: main training loop
# 153: radius optimization
# 267: optimize flattening/reconstruction pair per layer
# 376: automatically finding intrinsic dimension

# *---------------------------------------------------------*

import numpy as np
import torch

# used for optimizing over Stiefel
import geoopt

from .modules import flatnet_nn

from tqdm import trange

# for saving gifs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os
import atexit

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# delete temporary files if we exit too early
def exit_handler():
    if os.path.exists('flatnet_gif_frames'):
        for file_name in os.listdir('flatnet_gif_frames'):
            os.remove(os.path.join('flatnet_gif_frames', file_name))
        os.rmdir('flatnet_gif_frames')

# ******************************************************************************
# This is the primary script for the curvature compression algorithm.
# Input: data matrix X of shape (n,D), where D is the embedding dimension and
# n is the number of data points;
# d_desired is the desired dimension to flatten X onto

# Output: a neural network f: R^D -> R^d, where d is the intrinsic dimension of
# the data manifold X is drawn from.

# gamma_0 is the starting value of the "inverse neighborhood size"
# --- (the smaller gamma_0 is, the larger the neighborhood size is)

# main training loop
def train(X,
       n_stop_to_converge=10,  # how many times of no progress do we call convergence?
       n_iter=150,  # number of flattening steps to perform
       n_iter_inner=1000,  # how many max steps for inner optimization of U, V
       thres_recon=1e-4,  # threshold for reconstruction loss being good enough
       alpha_max=0.5,  # global parameter for kernel
       r_dimcheck_coeff=0.15,  # # radius for checking dimension.
       max_error_dimcheck_ratio=0.3,  # max l0 error with respect to r_dimcheck to stop dimension search
       r_min_coeff=0.1,  # minimum allowed radius for each flattening
       r_max_coeff=1.1,  # maximum allowed radius for each flattening
       r_step_min_coeff=1.0,  # max steps to take when finding biggest r
       n_iter_rsearch=1,  # max steps to take when finding biggest r
       save_gif = False, # whether to save gif of flattening process. currently only works for 2D data
       ):
    N, D = X.shape
    # needed for dist-to-gamma conversion
    log2 = float(np.log(2))

    # if save gif is set to true but data is not 2D, throw warning but continue running with save_gif = False
    if save_gif and not (D == 2 or D == 3):
        print("Warning: save_gif is set to True but data is not 2D or 3D. Setting save_gif to False")
        save_gif = False
    #######	## HYPERPARAMETERS ####
    ##############################

    converge_counter = 0

    # used to define hyperparameters
    EDM_X = torch.cdist(X, X, p=2)
    edm_max = EDM_X.max()
    edm_min = EDM_X[EDM_X > 0].min()

    # radius for checking dimension. Should be as small as possible,
    # but big enough that at every point there's at least one sample
    # along every intrinsic dimension
    r_dimcheck = r_dimcheck_coeff * edm_max
    # minimum allowed radius for each flattening
    # want this to be relatively larger to converge to flat
    # representation faster
    r_min = r_min_coeff * edm_max
    # maximum allowed radius for each flattening
    r_max = r_max_coeff * edm_max
    # 2nd radius to check for secant optimization
    r_midpoint = 0.5 * (r_min + r_max)
    # minimum step size for determining optimal radius
    r_step_min = r_step_min_coeff * edm_min

    # #################### TRACKING VARIABLES ####################
    # track previous intrinsic dimension to avoid redundant searches
    d_prev = 1

    ############# INIT GLOBAL VARIABLES##########
    # encoder network
    f = flatnet_nn.FlatteningNetwork()
    # decoder network
    g = flatnet_nn.FlatteningNetwork()
    Z = X.clone()

    ################### ANIMATION SAVING #################

    if save_gif:
        # print out to user that we are creating a folder to save frames temporarily
        print("NOTE: creating 'flatnet_gif_frames' directory to save frames temporarily. This directory will be deleted after the flattening process is complete and the gif is rendered.")
        # Ensure the 'flatnet_gif_frames' directory does not already exist
        if os.path.exists('flatnet_gif_frames'):
            raise Exception("The 'flatnet_gif_frames' directory already exists; to ensure this script does not damage your local files, please eiehter delete flatnet_gif_frames or move this script to a new directory and try again.")

        # Create the 'flatnet_gif_frames' directory
        os.makedirs('flatnet_gif_frames')

        # now if we exit, we need to delete the temp files
        atexit.register(exit_handler)

        # create array to store frames
        frames = []


    print(f'\nStart layer-wise training')
    # ################ MAIN LOOP #########################
    with trange(n_iter, unit="iters") as pbar:
        for j in pbar:
            # if j % 20 == 0:
            # 	plt.scatter(Z[:,0].detach().numpy(), Z[:,1].detach().numpy())
            # 	plt.show()

            # STEP 0: stochastically choose center of the neighborhood to
            # flatten and reconstruct
            choice = torch.randint(N, (1,))
            z_c = Z[choice, :]

            # STEP 1: find minimal dimension d we can flatten neighborhood
            # to and still be able to reconstruct

            # note d is implicitly returned, as U, V are of shape (D, d)
            U, loss_rdimcheck = find_d(Z, z_c, r_dimcheck, n_iter_inner, d_prev, max_error=thres_recon, max_error_dimcheck_ratio=max_error_dimcheck_ratio)

            # STEP 2: use secant method to find maximal radius that achieves
            # desired reconstruction loss

            # get needed second observation
            U, V, loss_rmidpoint = opt_UV(Z, z_c, U, n_iter_inner, r=r_midpoint)

            # radius optimization
            # begin secant method (note we use log loss for numerical reasons)
            log_thres_recon = torch.log(torch.Tensor([thres_recon]))
            r_m2 = r_dimcheck
            f_m2 = torch.log(loss_rdimcheck) - log_thres_recon
            r_m1 = (r_min + r_max) / 2
            f_m1 = torch.log(loss_rmidpoint) - log_thres_recon

            for _ in range(n_iter_rsearch):

                # threshold denominator for numerical stability
                f_diff = f_m1 - f_m2
                if torch.abs(f_diff) < 1e-6:
                    if f_diff >= 0:
                        f_diff = 1e-6
                    else:
                        f_diff = -1e-6

                r = r_m1 - (r_m1 - r_m2) / f_diff * f_m1

                # if we reach either boundary, threshold and exit
                if r < r_min:
                    r = r_min
                elif r > r_max:
                    r = r_max

                U, V, loss_r = opt_UV(Z, z_c, U, n_iter_inner, r=r)
                f_r = torch.log(loss_r) - log_thres_recon

                r_m2 = r_m1.clone()
                f_m2 = f_m1.clone()
                r_m1 = r.clone()
                f_m1 = f_r.clone()

                # stopping condition
                if torch.abs(r_m1 - r_m2) < r_step_min:
                    break

            # STEP 3: line search for biggest alpha that gets us to desired fidelity
            if loss_r.item() == 0.0:
                alpha = float(alpha_max)
            else:
                alpha = float(min(alpha_max, np.sqrt(thres_recon / loss_r.item())))
            # STEP 4: add layer to network
            Z = Z.detach()
            U = U.detach().clone()
            V = V.detach().clone()

            gamma = float(np.log(2)) / (r.item() ** 2)
            kernel_pre = torch.exp(-gamma * (Z - z_c).pow(2).sum(dim=1, keepdim=True))
            z_mu_local = (Z * kernel_pre).sum(dim=0, keepdim=True) / kernel_pre.sum()

            
            f_layer = flatnet_nn.FLayer(U, z_mu_local, gamma, alpha)
            f_layer.requires_grad_(False)
            g_layer = flatnet_nn.GLayer(U, V, z_mu_local, z_c, gamma, alpha)
            g_layer.requires_grad_(False)

            # test for convergence
            Z_new = f_layer(Z)

            # # add centering and normalization to manifold
            # z_mu = Z_new.mean(dim=0, keepdim=True)
            # # normalize by max norm of features
            # z_norm = (Z_new - z_mu).norm(dim=1).max()

            # f_layer.set_normalization(z_mu, z_norm)
            # g_layer.set_normalization(z_mu, z_norm)

            # # apply normalization forward
            # Z_new = (Z_new - z_mu) / z_norm
            
            # check for convergence
            if (Z_new - Z).pow(2).mean().sqrt() < 5e-5:
                # if we don't make any progress, don't add layer. However, we only
                # count convergence once radius is at its maximum
                if r.item() == r_max:
                    converge_counter += 1
                    if converge_counter >= n_stop_to_converge:
                        break
                else:
                    converge_counter = 0
            else:
                converge_counter = 0
                f.add_operation(f_layer)
                g.add_operation(g_layer)
                d_prev = U.shape[1]
                # only update representation if we add the layer
                Z = Z_new.clone()

                # save gif frame
                if save_gif:
                    if D == 2:
                        fig, ax = plt.subplots()
                        ax.scatter(Z[:, 0].detach().numpy(), Z[:, 1].detach().numpy())
                        ax.axis('off')
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                    elif D == 3:
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.axis('off')
                        ax.scatter(Z[:, 0].detach().numpy(), Z[:, 1].detach().numpy(), Z[:, 2].detach().numpy())
                        
                    plt.savefig(f"flatnet_gif_frames/frame_{j}.png")
                    plt.close()
                    # save frame
                    frames.append(imageio.imread(f"flatnet_gif_frames/frame_{j}.png"))

            # with torch.no_grad():
            # 	recon_loss = 0.5*(g(Z) - X).pow(2).mean()
            pbar.set_postfix({"local_recon": loss_r.item(), \
                              "d": U.shape[1], "r_ratio": (r / r_max).item(), "alpha": alpha})

    # final gif processing
    if save_gif:
        # save gif to be 6s long. note duration is duration
        # of each frame in ms
        imageio.mimsave('flatnet_flow.gif', frames, duration=6000 / len(frames))
        
        # delete auxillery files
        for file_name in os.listdir('flatnet_gif_frames'):
            os.remove(os.path.join('flatnet_gif_frames', file_name))
        os.rmdir('flatnet_gif_frames')

    print(f'Start training decoder globally')
    
    f.requires_grad_(False)
    for i in range(g.layer_count):
        for name,param in g.network[i].named_parameters():
            param.requires_grad_(False)
            if name == 'V':
                param.requires_grad_(True)

    optimizer = torch.optim.Adam(g.parameters(), lr=0.01)

    with trange(100, unit="iters") as pbar:
        for i in pbar:
            optimizer.zero_grad()
            X_hat = g(f(X))
            # breakpoint()
            loss = torch.nn.functional.mse_loss(X_hat, X,reduction='mean')
            loss.backward()
            # # add loss to tqdm
            # tqdm.tqdm.write(f'loss: {loss.item()}')

            pbar.set_postfix({"global_recon": loss.item()})
            optimizer.step()
            # if loss.item() < 1e-5:
            #     break

    return f, g


# ################## HELPER METHODS #####################

# optimize flattening/reconstruction pair per layer
def opt_UV(Z, z_c, U_0, n_iter_inner, r=-1, kernel=-1):
    D, d = U_0.shape
    N = Z.shape[0]
    # TESTING OPTION: whether or not to use cross terms of the
    # second fundamental form for reconstruction
    # NOTE: ALSO NEED TO CHANGE IN flatnet_nn.py IF CHANGING
    use_cross_terms = True

    # initialize geoopt manifold object
    stiefel = geoopt.manifolds.Stiefel()
    # make parameter for geoopt
    with torch.no_grad():
        U = geoopt.ManifoldParameter(U_0, manifold=stiefel).proj_()
    # optimize U, V
    # U = torch.nn.Parameter(U_0.clone())
    # U.requires_grad = True

    # opt_U = optim.SGD([U], lr=0.1)
    opt_U = geoopt.optim.RiemannianAdam([U], lr=1)

    # must specify either r or kernel
    if type(r) != torch.Tensor and type(kernel) != torch.Tensor:
        raise ValueError('Must specify either r or kernel')

    # init kernel
    if type(kernel) != torch.Tensor:
        gamma = float(np.log(2)) / (r ** 2)
        kernel_pre = torch.exp(-gamma * (Z - z_c).pow(2).sum(dim=1, keepdim=True))
        z_mu = (Z * kernel_pre).sum(dim=0, keepdim=True) / kernel_pre.sum()
        kernel = torch.exp(-gamma * (Z - z_mu).pow(2).sum(dim=1, keepdim=True))

    # find sub-selection for nonzero kernel elements
    n_idx = kernel.flatten() > 1e-6
    with torch.no_grad():
        Z_train = Z[n_idx, :].detach().clone()
        kernel_train = kernel[n_idx].detach().clone()

    for _ in range(n_iter_inner):
        U_old = U.data.clone()
        opt_U.zero_grad()
        # opt_V.zero_grad()
        # opt_alpha.zero_grad()

        coord = (Z_train - z_c) @ U
        Z_perp = (Z_train - z_c) - coord @ U.T

        if not use_cross_terms:

            coord2 = coord.pow(2)
            A = coord2 * kernel_train
            b = Z_perp * kernel_train

            # least squares solution for V, note automatically orthogonal to U
            with torch.no_grad():
                # V = (torch.linalg.pinv(A, rtol=1e-6)@b).T
                V = torch.linalg.lstsq(A, b, rcond=1e-4).solution.T

            loss = 0.5 * (kernel_train * (Z_perp - coord2 @ V.T)).pow(2).mean()

        else:
            # input coordinates for the exponential map hessian (second
            # fundamental form)

            # multiply batch of matrices
            # output is a tensor of shape (N,d,d)
            # H_input = torch.bmm(coord.reshape((N,d,1)), coord.reshape((N,1,d)))
            H_input = torch.bmm(coord.unsqueeze(2), coord.unsqueeze(1))
            # since tensor is symmetric, we only need to keep upper diag terms
            # output is a tensor of shape (N, d(d+1)/2)
            idx_triu = torch.triu_indices(d, d)
            H_input = H_input[:, idx_triu[0, :], idx_triu[1, :]]

            # construct A, b matrices for least squares
            A = H_input * kernel_train
            b = Z_perp * kernel_train

            # least squares solution for V, note automatically orthogonal to U
            # output is of shape (D, d(d+1)/2)
            with torch.no_grad():
                # V = ((A.T@A).inverse() @ (A.T@b)).T
                V = torch.linalg.lstsq(A, b, rcond=1e-4).solution.T

            loss = 0.5 * (kernel_train * (Z_perp - H_input @ V.T)).pow(2).mean()

        # loss = (U).pow(2).mean()
        loss.backward()

        opt_U.step()

        with torch.no_grad():

            # # project onto Stiefel manifold
            # if U.data.shape[1] == 1:
            # 	U.data = U.data / torch.norm(U.data, p=2)
            # else:
            # 	U_svd, S_svd, Vh_svd = torch.linalg.svd(U.data, full_matrices=False)
            # 	U.data = U_svd@Vh_svd

            step_size = (U.data - U_old).pow(2).mean().sqrt()
            U_old = U.data.clone()

            if step_size < 1e-5:
                break
    # if i >= n_iter_inner - 1:
    # 	print('Warning: U did not converge')
    return U.detach().data, V.detach().data, loss.detach()


# automatically finding intrinsic dimension
def find_d(Z, z_c, r_dimcheck, n_iter_inner, d_prev, max_error, max_error_dimcheck_ratio=0.3):
    # We find the minimial d by iteratively fitting a model
    # for some size d, then increase d and repeat if the max
    # reconstruction error is too large

    max_error = max_error_dimcheck_ratio * r_dimcheck

    N, D = Z.shape
    # init
    U_0 = torch.randn(D, d_prev)
    U_0 = torch.linalg.qr(U_0)[0]

    # TRACKING VARS
    was_decreasing = False

    # init tracking variable
    # max_error = max_error_ratio*r_dimcheck

    # note kernel will stay the same for all d
    gamma = float(np.log(2)) / (r_dimcheck ** 2)
    kernel_pre = torch.exp(-gamma * (Z - z_c).pow(2).sum(dim=1, keepdim=True))
    z_mu = (Z * kernel_pre).sum(dim=0, keepdim=True) / kernel_pre.sum()
    kernel = torch.exp(-gamma * (Z - z_mu).pow(2).sum(dim=1, keepdim=True))

    # need to track U_prev in case of decreasing
    U_prev = U_0.clone()

    # note this loop iterates a maximum of D times
    for i in range(D):
        U, V, loss = opt_UV(Z, z_c, U_0, n_iter_inner, kernel=kernel)

        # compute max error, if too large, increase d
        coord = (Z - z_c) @ U
        Z_perp = (Z - z_c) - coord @ U.T
        # NOTE THIS NEEDS TO CHANGE IF DOING DIAG HESSIAN
        H_input = torch.bmm(coord.unsqueeze(2), coord.unsqueeze(1))
        # since tensor is symmetric, we only need to keep upper diag terms
        # output is a tensor of shape (N, d(d+1)/2)
        d = U.shape[1]
        idx_triu = torch.triu_indices(d, d)
        H_input = H_input[:, idx_triu[0, :], idx_triu[1, :]]
        l0_error = (kernel * (Z_perp - H_input @ V.T)).norm(dim=1).max()
        # l2_error = 0.5*(kernel * (Z_perp - H_input @ V.T)).norm(dim=1).pow(2).mean()
        # if achieved desired loss, reduce dimension
        # print(f'd: {d}, error: {l0_error}')
        if l0_error <= max_error:
            # if we were increasing but then achieved desired loss we have converged
            if (not was_decreasing and i > 0) or U.shape[1] == 1:
                break

            # otherwise, continue to reduce dimension
            was_decreasing = True
            # remove column of U lease correlated with V
            idx_min = ((Z - z_c) @ U).abs().mean(dim=0).argmin()
            U_0 = torch.cat((U[:, :idx_min], U[:, idx_min + 1:]), dim=1)
            U_prev = U.clone()

        else:
            # if we were decreasing but then achieved desired loss we have converged
            if was_decreasing or U.shape[1] == D:
                U = U_prev
                break
            U_new = torch.randn(D, 1)
            U_new = U_new - U @ (U.T @ U_new)
            U_new = U_new / torch.norm(U_new, p=2)
            U_0 = torch.cat((U, U_new), dim=1).clone()

    return U.detach().data, loss.detach().data
