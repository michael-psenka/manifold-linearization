from statistics import mean, stdev
import torch
from models.vae import train_vanilla_vae, train_beta_vae, train_factor_vae
from tools.gp_manifold_generator import sample_points
from cc import cc
import matplotlib.pyplot as plt


def mse(X, F, G):
    N, D = X.shape
    return ((torch.linalg.norm(X - G(F(X))) ** 2) / N).detach().item()

N_trials = 3

N = 2000
D = 100
d_init = 4
d_max = 20
d_skip = 4

d_list = []
cc_recon_mean = []
vae_recon_mean = []
bvae_recon_mean = []
fvae_recon_mean = []
cc_recon_std = []
vae_recon_std = []
bvae_recon_std = []
fvae_recon_std = []

for d in range(d_init, d_max+1, d_skip):
    d_list.append(d)
    cc_recon = []
    vae_recon = []
    bvae_recon = []
    fvae_recon = []
    for i in range(N_trials):
        X, _, _ = sample_points(N, D, d, [1.0 for _ in range(D)])

        F, G = cc(X)
        cc_recon.append(mse(X, F, G))
        X = X.detach()

        F, G = train_vanilla_vae(X, d_z=d, d_latent=D)
        vae_recon.append(mse(X, F, G))
        X = X.detach()

        F, G = train_beta_vae(X, d_z=d, d_latent=D)
        bvae_recon.append(mse(X, F, G))
        X = X.detach()

        F, G = train_factor_vae(X, d_z=d, d_latent=D)
        fvae_recon.append(mse(X, F, G))
        X = X.detach()

    cc_recon_mean.append(mean(cc_recon))
    cc_recon_std.append(stdev(cc_recon))

    vae_recon_mean.append(mean(vae_recon))
    vae_recon_std.append(stdev(vae_recon))

    bvae_recon_mean.append(mean(bvae_recon))
    bvae_recon_std.append(stdev(bvae_recon))

    fvae_recon_mean.append(mean(fvae_recon))
    fvae_recon_std.append(stdev(fvae_recon))

plt.title("$\mathbb{E}[\|x - \hat{x}\|_{2}^{2}]$, $D = " + str(D) + "$")
plt.xlabel("$d$")
plt.plot(d_list, cc_recon_mean, label="CCNet", color="C0")
plt.plot(d_list, vae_recon_mean, label="VAE", color="C1")
plt.plot(d_list, bvae_recon_mean, label="$\\beta$-VAE", color="C2")
plt.plot(d_list, fvae_recon_mean, label="FactorVAE", color="C3")
plt.fill_between(d_list,
                 [cc_recon_mean[i] - cc_recon_std[i] for i in range(len(cc_recon_mean))],
                 [cc_recon_mean[i] + cc_recon_std[i] for i in range(len(cc_recon_mean))],
                 color="C0", alpha=0.1)
plt.fill_between(d_list,
                 [vae_recon_mean[i] - vae_recon_std[i] for i in range(len(vae_recon_mean))],
                 [vae_recon_mean[i] + vae_recon_std[i] for i in range(len(vae_recon_mean))],
                 color="C1", alpha=0.1)
plt.fill_between(d_list,
                 [bvae_recon_mean[i] - bvae_recon_std[i] for i in range(len(bvae_recon_mean))],
                 [bvae_recon_mean[i] + bvae_recon_std[i] for i in range(len(bvae_recon_mean))],
                 color="C2", alpha=0.1)
plt.fill_between(d_list,
                 [fvae_recon_mean[i] - fvae_recon_std[i] for i in range(len(fvae_recon_mean))],
                 [fvae_recon_mean[i] + fvae_recon_std[i] for i in range(len(fvae_recon_mean))],
                 color="C3", alpha=0.1)
plt.legend()
plt.savefig("reconstruction_comparison.jpg")
plt.close()
