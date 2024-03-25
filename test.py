import torch
import flatnet
import matplotlib.pyplot as plt

# create a circle dataset


t = torch.linspace(0, 4, 100)
y = torch.where(t < 3, torch.sin(2*t),1/4*t**2)

# format dataset of N points of dimension D as (N, D) matrix
X = torch.stack([t, y], dim=1)
# add noise
X = X + 0.02*torch.randn(*X.shape)

# normalize data
X = (X - X.mean(dim=0)) / X.std(dim=0)

# train encoder f and decoder g, then save a gif of the
# manifold evolution through the constructed layers of f
f, g = flatnet.train(X, save_gif=True,n_iter=500)

# plot the original and reconstructed data
Z = f(X)
X_hat = g(Z)
# plot the results
fig, ax = plt.subplots(1, 2)
ax[0].plot(X[:, 0], X[:, 1], 'o')
ax[0].plot(X_hat[:, 0], X_hat[:, 1], 'o')
ax[0].set_title('Original and Reconstructed Data')
ax[1].plot(Z[:, 0], Z[:, 1], 'o')
ax[1].set_title('Latent Space')
plt.show()
