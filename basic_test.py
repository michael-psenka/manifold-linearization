import torch
import flatnet
import matplotlib.pyplot as plt
import copy
import tqdm
from flatnet.modules import flatnet_nn
import manifold_curvature as mc

# create sine wave dataset
t = torch.linspace(0*torch.pi, 2*torch.pi, 50)
y = torch.sin(t)

# format dataset of N points of dimension D as (N, D) matrix
X = torch.stack([t, y], dim=1)
# add noise
X = X + 0.02*torch.randn(*X.shape)

# normalize data
X = X - X.mean(dim=0)
X = X / X.norm(dim=1).max()

sinMC = mc.ManifoldCurvature()
f,g=sinMC.fit(X, latent_dim=1,n_iter=100)
Z = f(X)
d = Z.shape[1]
# interpolate two samples between each two points in Z
Z1 = torch.empty(0)
inter_num = 2
for i in range(1, len(Z)):
    for j in range(inter_num):
        Z1 = torch.cat((Z1, Z[i-1] + (Z[i] - Z[i-1]) * (j+1) / (inter_num+1)))
# breakpoint()
Z1 = Z1.view(-1, d)
Z1 = torch.cat((Z, Z1))

X1_hat = g(Z1)
X_hat = g(Z)
# middle_point = X[len(X)//2]
# tangent_space = sinMC.tangent_space(middle_point)
# hessian = sinMC.hessian(middle_point)
# print("Tangent space of ", middle_point, " is ", tangent_space)
# print("Hessian of ", middle_point, " is ", hessian)

curvature,h_all,j_all = sinMC.curvature(X, only_jacobian=True)
print("Curvature of sin wave dataset: ")
print(curvature)

j_all = j_all.detach().numpy()

X_hat = X_hat.detach().numpy()
X1_hat = X1_hat.detach().numpy()
h_all = h_all.detach().numpy()
# TODO: PLOT CURVATURE VECTOR
# plt.quiver(*X_hat.T, h_all[:, 0], h_all[:, 1], color='g')
plt.quiver(*X_hat.T, j_all[:, 0,0], j_all[:, 1,0], color='r', scale=10, width=0.0015)
# smaller points
plt.scatter(X1_hat[:, 0], X1_hat[:, 1], s=0.5)
# plt.scatter(X[:, 0], X[:, 1])
plt.axis('on')
plt.savefig('curvature_vector.png', dpi=800)
plt.clf()





# plot the sine wave dataset
plt.scatter(X[:, 0], X[:, 1])
plt.savefig('sine_wave_dataset.png')
plt.clf()

# plot the curvature of the sine wave dataset
plt.scatter(X[:, 0], curvature)
plt.savefig('sine_wave_curvature.png')
plt.clf()