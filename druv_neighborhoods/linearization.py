import random
import torch
from tqdm import tqdm
from neighborhood_selection import neighborhood_selection


def find_linearization_function(X: torch.Tensor, d: int, k_min: int, k_max: int, eta: float, lr: float,
                                n_layers: int, n_iters: int):
    neighborhoods = neighborhood_selection(X, d, k_min, k_max, eta)
    n, m = X.shape
    layers = []
    for _ in range(n_layers):
        layers.extend([torch.nn.Linear(m, m), torch.nn.ReLU()])
    layers.append(torch.nn.Linear(m, m))
    F = torch.nn.Sequential(*layers)
    optim = torch.optim.SGD(F.parameters(), lr=lr)

    # Training algorithm: for a certain number of iterations, pick a neighborhood and flatten it
    for idx in tqdm(range(n_iters)):
        i = random.choice(range(n))
        X_i = neighborhoods[i]
        FX_i = F(X_i)
        Fx_i_bar = torch.mean(FX_i, dim=0)
        FX_i_centered = FX_i - Fx_i_bar
        U, S, Vh = torch.linalg.svd(FX_i_centered)
        Q_i_T = Vh[:d]
        loss = torch.linalg.norm(FX_i - (Fx_i_bar + FX_i_centered @ Q_i_T.T @ Q_i_T)) ** 2
        loss.backward()
        optim.step()
        optim.zero_grad()
    F.eval()
    return F


__all__ = ["find_linearization_function"]
