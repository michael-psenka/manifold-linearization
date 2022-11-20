import torch
import time


def dimension_estimation(train_fn, X: torch.Tensor, eps: float):
    # Trains models with increasing latent dimension d_z until the reconstruction loss is below eps.
    # Outputs dimension estimate and timer.
    start_time = time.time()
    d_z = 0
    train_error_needs_improvement = True
    while train_error_needs_improvement:
        d_z += 1
        F, G = train_fn(X, d_z=d_z)
        Xhat = F(G(X))
        err = torch.linalg.norm(X - Xhat)
        if err < eps:
            end_time = time.time()
            print("Dimension estimate:", d_z)
            print("Time:", end_time - start_time)

