import json
import os
import torch
import numpy as np
from tools.gp_manifold_generator import sample_points
from models.vae import train_vanilla_vae, train_beta_vae, train_factor_vae
from flatnet import train
import matplotlib.pyplot as plt

def distance_metric_ratio(X, P, F, eps = 1e-12):
    L = F(X)
    ED = torch.cdist(L, L) ** 2 + eps
    RD = torch.cdist(P, P) ** 2 + eps

    return torch.divide(ED,RD)

def generate_plot(X, filename):
    results_filename = f"edm_experiment/{filename}.png"
    X_np = X.detach().numpy()
    heatmap = plt.pcolor(X_np, shading = 'auto', vmin = 0, vmax=1, cmap='Greys')
    plt.colorbar(orientation='vertical')
    plt.savefig(results_filename)
    plt.clf()


N = 3000
D = 100
d = 4
stopping_time = [40, 60, 80, 100, 120]

X, P, _ = sample_points(N, D, d, [1.0 for _ in range(D)])

print('generated manifold')



flatnet_F, flatnet_G = train(X)
flatnet_edm = distance_metric_ratio(X, P, flatnet_F)

generate_plot(flatnet_edm, f'flatnet_N{N}_D{D}_d{d}')

for epochs in stopping_time:
    print(f"running epoch {epochs}")

    vae_F, vae_G = train_vanilla_vae(X, num_epochs=epochs)
    bvae_F, bvae_G = train_beta_vae(X, num_epochs=epochs)
    fvae_F, fvae_G = train_factor_vae(X, num_epochs=epochs)

    vae_edm = distance_metric_ratio(X, P, vae_F)
    bvae_edm = distance_metric_ratio(X, P, bvae_F)
    fvae_edm = distance_metric_ratio(X, P, fvae_F)

    generate_plot(vae_edm, f'vae_N{N}_D{D}_d{d}_epochs{epochs}')
    generate_plot(bvae_edm, f'bvae_N{N}_D{D}_d{d}_epochs{epochs}')
    generate_plot(fvae_edm, f'fvae_N{N}_D{D}_d{d}_epochs{epochs}')



    






