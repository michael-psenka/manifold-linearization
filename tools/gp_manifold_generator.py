from typing import List
import numpy as np
import torch
import scipy.linalg


def sample_points(N: int, D: int, d: int, L: List[float]):
    X = torch.zeros(size=(N, D))  # Extrinsic coordinates
    P = torch.rand(size=(N, d))  # Intrinsic coordinates
    rho = torch.cdist(P, P) ** 2  # Pairwise distances
    for c in range(D):
        Z = torch.randn(size=(N, ))
        Q = ((L[c] ** 2) / D) * torch.exp(-rho / 2)
        m = Q.detach().cpu().numpy().astype(np.float_)
        Q_sqrt = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(Q)
        X[:, c] = Q_sqrt @ Z
    return X, P, d

