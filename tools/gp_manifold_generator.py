from typing import List

import torch


def sample_points(N: int, D: int, d: int, L: List[float]):
    X = torch.zeros(size=(N, D))  # Extrinsic coordinates
    P = torch.rand(size=(N, d))  # Intrinsic coordinates
    rho = torch.cdist(P, P) ** 2  # Pairwise distances
    ell = sum(L)
    for c in range(D):
        Z = torch.randn(size=(N, ))
        Q = ((ell ** 2) / D) * torch.exp(-rho / 2)
        Lmbda, V = torch.linalg.eigh(Q)
        Q_sqrt = V @ torch.diag(Lmbda ** 0.5) @ V.T
        X[:, c] = Q_sqrt @ Z
    return X, P, d

