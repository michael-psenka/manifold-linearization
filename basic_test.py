import torch
import flatnet
import matplotlib.pyplot as plt

# create sine wave dataset
t = torch.linspace(0, 2*torch.pi, 40)
y = torch.sin(t)

y1= torch.sin(t)+2

# format dataset of N points of dimension D as (N, D) matrix
X = torch.stack([t, y], dim=1)
X1 = torch.stack([t, y1], dim=1)
X = torch.cat([X, X1], dim=0)
# add noise
X = X + 0.02*torch.randn(*X.shape)

# normalize data
X = X - X.mean(dim=0)
X = X / X.norm(dim=1).max()

# train encoder f and decoder g, then save a gif of the
# manifold evolution through the constructed layers of f
flatnet_this = flatnet.train(X, n_iter=10, save_gif=True)
# reconstruct the data
Z = flatnet_this.encode(X)
X_hat = flatnet_this.decode(Z)
# plot the results
fig, ax = plt.subplots(1, 2)
ax[0].plot(X[:, 0], X[:, 1], 'o',label='Original Data')
ax[0].plot(X_hat[:, 0], X_hat[:, 1], 'o',label='Reconstructed Data')
ax[0].set_title('Original and Reconstructed Data')
ax[0].legend()
ax[1].plot(Z[:, 0], Z[:, 1], 'o')
ax[1].set_title('Latent Space')
plt.savefig('basic_test.png')


