import json
import os
import torch
import numpy as np
from tools.gp_manifold_generator import sample_points
from models.vae import train_vanilla_vae, train_beta_vae, train_factor_vae
from cc import cc

def distance_metric_ratio(X, P, F, eps = 1e-12):
    L = F(X)
    ED = torch.cdist(L, L) ** 2 + eps
    RD = torch.cdist(P, P) ** 2 + eps

    return torch.min(ED/RD), torch.max(ED/RD)

N_trials = 5

N = 2000
D = 100
d = 4
stopping_time = [40, 60, 80, 100, 120]

for i in range(N_trials):
    X, P, _ = sample_points(N, D, d, [1.0 for _ in range(D)])
    print("generated manifold")
    results = {}
    results_filename = f"edm_experiment/results_N{N}_D{D}_d{d}_i{i}.txt"
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)     
    cc_F, cc_G = cc(X)
    cc_min, cc_max = distance_metric_ratio(X, P, cc_F)
    results["cc_min"] = cc_min
    results["cc_max_"] = cc_max

    for epochs in stopping_time:
        print(f"running epoch {epochs}")

        vae_F, vae_G = train_vanilla_vae(X, num_epochs=epochs)
        bvae_F, bvae_G = train_beta_vae(X, num_epochs=epochs)
        fvae_F, fvae_G = train_factor_vae(X, num_epochs=epochs)

        vae_min, vae_max = distance_metric_ratio(X, P, vae_F)
        bvae_min, bvae_max = distance_metric_ratio(X, P, bvae_F)
        fvae_min, fvae_max = distance_metric_ratio(X, P, fvae_F)

        results[f"vae_min_{epochs}"] = vae_min
        results[f"vae_max_{epochs}"] = vae_max
        results[f"bvae_min_{epochs}"] = bvae_min
        results[f"bvae_max{epochs}"] = bvae_max
        results[f"fvae_min_{epochs}"] = fvae_min
        results[f"fvae_max_{epochs}"] = fvae_max

    if not os.path.exists(results_filename):
        torch.save(results, results_filename)
    






