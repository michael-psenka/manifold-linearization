from matplotlib import axes
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
import torch

from cc import cc

# circle dataset
n = 100
r = 10
x = np.linspace(0, 2 * np.pi, n)

X = torch.zeros((n*n, 3))
for i in range(n):
    for j in range(n):
        X[n*i + j] = torch.tensor([x[i], r * np.cos(x[j]), abs(r * np.sin(x[j]))])

print(X)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0].detach().numpy(), X[:, 1].detach().numpy(), X[:, 2].detach().numpy())
plt.show()

#swiss roll
# X = torch.from_numpy(make_swiss_roll(1000, noise = 0.1)[0]).type(torch.FloatTensor)

# print(X.shape)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(X[:, 0].detach().numpy(), X[:, 1].detach().numpy(), X[:, 2].detach().numpy())
# plt.show()


F, _ = cc(X)
Z = F(X)
print(torch.norm(Z - X))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Z[:, 0].detach().numpy(), Z[:, 1].detach().numpy(), Z[:, 2].detach().numpy())
plt.show()