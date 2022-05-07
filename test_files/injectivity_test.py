import typing
import torch
import pytorch_lightning as pl


def injectivity_distance_ratio(
        F: typing.Callable[[torch.Tensor], torch.Tensor],  # F: R^d -> R^D
        X: torch.Tensor,  # (d, n)
        eps: float = 1e-10
):
    """
    Measures injectivity by comparing ||f(x^i) - f(x^j)|| / ||x^i - x^j||.

    :param F: The encoder function R^d -> R^D, also broadcasts R^(n x d) -> R^(n x D).
    :param X: The data, of shape (d, n)
    :param eps: Some tiny divide-by-zero protection.
    :return: min_(i != j) ||f(x^i) - f(x^j)|| / ||x^i - x^j||, a measure of injectivity.
    """
    Zt = F(X.T)
    X_distances = torch.cdist(X.T, X.T) + eps
    Z_distances = torch.cdist(Zt, Zt) + eps
    return torch.min(Z_distances / X_distances)


class TwoLayerRegressionNN(pl.LightningModule):
    def __init__(self, d_in: int, d_latent: int, d_out: int):
        super(TwoLayerRegressionNN, self).__init__()
        self.fc1 = torch.nn.Linear(d_in, d_latent)
        self.fc2 = torch.nn.Linear(d_latent, d_out)

    def forward(self, x: torch.Tensor):
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))


def train_twolayer_regression_nn(X: torch.Tensor, Z: torch.Tensor, d_latent: int = 100, lmbda: float = 500,
                                 lr: float = 1e-3, epochs: int = 10, batch_size: int = 20):  # (d, n), (D, n)
    X_copy = torch.clone(X)
    Z_copy = torch.clone(Z)
    X_copy.detach()
    Z_copy.detach()

    X = X_copy.T  # (n, d)
    Z = Z_copy.T  # (n, D)

    n, d = X.shape
    D = Z.shape[1]

    nn = TwoLayerRegressionNN(d, d_latent, D)  # represents mapping R^d -> R^D
    optimizer = torch.optim.Adam(nn.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X, Z)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for idx, data in enumerate(dataloader):
            x_batch, z_batch = data  # (n_i, d), (n_i, D)

            optimizer.zero_grad()

            z_predicted = nn(x_batch)  # (n_i, D)

            mse = torch.linalg.norm(z_predicted - z_batch) ** 2
            # i know you can use weight decay but wanted to be more explicit
            l2_weights = sum(torch.linalg.norm(Ai) ** 2 for Ai in nn.parameters())
            loss = mse + lmbda * l2_weights
            loss.backward(retain_graph=True)

            optimizer.step()

        nn.eval()
        Z_predicted = nn(X)
        nn.train()
        print(f"Epoch {epoch}:")
        print(f"MSE = ", (torch.linalg.norm(Z_predicted - Z) ** 2) / n)
        print(f"Parameter weights = ", sum(torch.linalg.norm(Ai) ** 2 for Ai in nn.parameters()))
        print("\n")
    nn.eval()
    return nn


def injectivity_nn_weights(
        F: typing.Callable[[torch.Tensor], torch.Tensor],  # F: R^d -> R^D
        X: torch.Tensor,  # (d, n)
        d_latent: int = 100,
        lmbda: float = 500
):
    Z = F(X.T).T  # (D, n)
    G_nn = train_twolayer_regression_nn(Z, X, d_latent, lmbda, epochs=1000)
    return sum(torch.linalg.norm(Ai) ** 2 for Ai in G_nn.parameters())
