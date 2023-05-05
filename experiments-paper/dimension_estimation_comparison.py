from statistics import mean, stdev, mode
import torch
from models.vae import train_vanilla_vae, train_beta_vae, train_factor_vae
from tools.gp_manifold_generator import sample_points
from flatnet import train
import matplotlib.pyplot as plt
import skdim


def flatnet_dim_estimation(F):
    return mode([layer.k for layer in F.network])


N_trials = 3

N = 2000
D = 100
d_init = 5
d_max = 20
d_skip = 5

d_list = []
flatnet_id_mean = []
mle_id_mean = []
tnn_id_mean = []
flatnet_id_std = []
mle_id_std = []
tnn_id_std = []

for d in range(d_init, d_max+1, d_skip):
    d_list.append(d)
    flatnet_id = []
    mle_id = []
    tnn_id = []
    for i in range(N_trials):
        X, _, _ = sample_points(N, D, d, [1.0 for _ in range(D)])

        F, G = train(X)
        flatnet_id.append(flatnet_dim_estimation(F))
        X = X.detach().numpy()

        mle_id.append(skdim.id.MLE().fit_transform(X))
        tnn_id.append(skdim.id.TwoNN().fit_transform(X))

    flatnet_id_mean.append(mean(flatnet_id))
    flatnet_id_std.append(stdev(flatnet_id))
    mle_id_mean.append(mean(mle_id))
    mle_id_std.append(stdev(mle_id))
    tnn_id_mean.append(mean(tnn_id))
    tnn_id_std.append(stdev(tnn_id))

plt.title(f"Estimated intrinsic dimension of data, $D = {D}$.")
plt.xlabel("$d$")
plt.plot(d_list, flatnet_id_mean, label="FlatNet", color="C0")
plt.plot(d_list, mle_id_mean, label="MLE", color="C1")
plt.plot(d_list, tnn_id_mean, label="TwoNN", color="C2")
plt.fill_between(d_list,
                 [flatnet_id_mean[i] - flatnet_id_std[i] for i in range(len(flatnet_id_mean))],
                 [flatnet_id_mean[i] + flatnet_id_std[i] for i in range(len(flatnet_id_mean))],
                 color="C0", alpha=0.1)
plt.fill_between(d_list,
                 [mle_id_mean[i] - mle_id_std[i] for i in range(len(mle_id_mean))],
                 [mle_id_mean[i] + mle_id_std[i] for i in range(len(mle_id_mean))],
                 color="C1", alpha=0.1)
plt.fill_between(d_list,
                 [tnn_id_mean[i] - tnn_id_std[i] for i in range(len(tnn_id_mean))],
                 [tnn_id_mean[i] + tnn_id_std[i] for i in range(len(tnn_id_mean))],
                 color="C2", alpha=0.1)
plt.plot([d_init, d_max], [d_init, d_max], color="C3", linestyle="dashed", label="true $d$")
plt.legend()
plt.savefig("id_estimation_comparison.jpg")
plt.close()
