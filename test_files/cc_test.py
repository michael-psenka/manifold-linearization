# %matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
# setting path
sys.path.append('../')
from tools.manifold_generator import Manifold


hi = Manifold(D=2, d=1, curvature=0.5, n_basis=10)
X = hi.generateSample(N=500, uniform=True)
plt.plot(X[0,:], X[1,:], '.')
plt.show()


# # dimension of data
# d = 2
# # number of samples
# n = 25
# # variance of noise
# eps2 = 0
# # determine if we should randomly rotate
# rotate=False


# # 2D manifold in 4D space

# # sqrt_n = 25
# # n = sqrt_n**2

# # Z = np.zeros((d,n))
# # for i in range(sqrt_n):
# # 	for j in range(sqrt_n):
# # 		x = (-0.5 + (i+1)/sqrt_n)*2*np.pi
# # 		y = (-0.5 + (j+1)/sqrt_n)*2*np.pi
# # 		Z[0,j + sqrt_n*i] = x
# # 		Z[1,j + sqrt_n*i] = y
# # 		Z[2,j + sqrt_n*i] = np.sin(4*x)
# # 		Z[3,j + sqrt_n*i] = np.sin(7*y)


# # # mean center
# # Z = Z - np.mean(Z,axis=1,keepdims=True)
# # # global normalization
# # Z = Z * n / np.sqrt(np.sum(np.power(Z, 2)))

# # plt.scatter(Z[0,:], Z[1,:])
# scale=2
# Z = np.zeros((d,n))
# for i in range(n):
# 	x = (-0.5 + (i+1)/n)*2*np.pi
# 	Z[0,i] = x
# 	# Z[1,i] = scale*np.sin(x)
# 	Z[1,i] = scale*np.sin(x) + scale/2*np.sin(3*x + 1.5)

# # add noise
# Z = Z + eps2*np.random.randn(d, n)


# # center and scale
# Z = Z - Z.mean(axis=1,keepdims=True)
# Z = Z * n / np.linalg.norm(Z, 'fro')

# # random rotation, preserves centering and scale
# if rotate:
# 	A = np.random.randn(d,d)
# 	u,s,vt = np.linalg.svd(A)
# 	Z = u@Z

# plt.scatter(Z[0,:], Z[1,:])

# u, s, vt = np.linalg.svd(Z)
# u_min = u[:,1]
# print(u_min)

# # scale to look nice on graph
# u_show = u_min * np.min(np.max(np.abs(Z), axis=1))/2

# plt.scatter(Z[0,:], Z[1,:])
# plt.quiver(0, 0, u_show[0], u_show[1],scale_units='xy', angles='xy',scale=1)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

# # # Curvature compression on manifold data
# # 
# # We now test our new proposed method, curvature compression. The function we theoretically want to optimize for is the following:
# # 
# # $$v_* = \argmin_{\|v\|_2 = 1} \max_{x, y \in Z \mid x \ne y} |\langle v, \frac{x-y}{\|x-y\|_2}\rangle|$$
# # $$= \argmin_{\|v\|_2 = 1} \|\langle v, \frac{x-y}{\|x-y\|_2}\rangle\|_\infty. $$
# # 
# # However, since the $L^\infty$ norm is hard to directly optimize, we settle for a smooth surrogate norm: the $L^4$-norm.

# # PyTorch model to optimize our custom loss
# class CurvatureElimination(nn.Module):

#     def __init__(self, X, u_0):
#         super(CurvatureElimination, self).__init__();
#         # data
#         self.X = X
#         # construct edm and gamma weight matrix
#         gram = X.T@X
#         edm = torch.diag(gram).reshape((n,1))@torch.ones((1,n)) \
#                 + torch.ones((n,1))@torch.diag(gram).reshape((1,n)) \
#                 - 2*gram

#         # diagonal weights don't matter, set to 1 as convention
#         self.gamma = torch.divide(torch.Tensor([1]),torch.eye(n) + edm)
#         # projection vec
#         self.u = nn.Parameter(u_0)

#     def forward(self):
#         U_stacked = torch.ones(self.X.shape)*self.u
#         A = (self.X.T @ U_stacked - U_stacked.T @ self.X)*self.gamma
#         return 0.5*A.pow(4).sum()
# # optimize via projected gradient descent

# u_0 = torch.randn((d,1))
# u_0 = u_0 / torch.sqrt(u_0.pow(2).sum())
# X = torch.Tensor(Z)

# cc = CurvatureElimination(X, u_0)
# opt = optim.SGD(cc.parameters(), lr=0.0001)

# for i in range(10000):
# 	cc.zero_grad()
# 	# forward call of LinFlow
# 	loss = cc()

# 	loss.backward()

# 	# compute Riemannian gradient
# 	egrad = cc.u.grad.detach()
# 	base = cc.u.data.detach()
# 	base = base / torch.sqrt(base.pow(2).sum())
# 	rgrad = egrad - base@base.t()@egrad
# 	# GD step
# 	opt.step()
# 	# renormalization step
# 	with torch.no_grad():
# 		cc.u.data = cc.u.data / torch.sqrt((cc.u.data).pow(2).sum());

# 	# determine if we have converged
# 	gradnorm = torch.linalg.norm(rgrad)
# 	if gradnorm < 5e-5:
# 		print(f'converged in {i} steps!')
# 		break
# 	if i%100 == 0:
# 		print(f'g step {i}: {gradnorm}')

# print('done!')

# # show learned compression direction
# u_cc = cc.u.data.detach().clone().numpy()

# print(f'learned compression direction: {u_cc.T}')
# # scale to look nice on graph
# u_show = u_cc * np.min(np.max(np.abs(Z), axis=1))/2

# plt.scatter(Z[0,:], Z[1,:])
# plt.quiver(0, 0, u_show[0], u_show[1],scale_units='xy', angles='xy',scale=1)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()