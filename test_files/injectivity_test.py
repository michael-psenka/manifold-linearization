import typing
import torch
import pytorch_lighting as pl


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


class TwoLayerRegressionNN(torch.nn.Module):
    def __init__(self, d_in: int, d_latent: int, d_out: int):
        super(TwoLayerRegressionNN, self).__init__()
        self.fc1 = torch.nn.Linear(d_in, d_latent)
        self.fc2 = torch.nn.Linear(d_latent, d_out)

    def forward(self, x: torch.Tensor):
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))

def train_twolayer_regression_nn(X: torch.Tensor, Z: torch.Tensor, d_latent: int, lmbda: float = 500,
                                 lr: float = 1e-3, epochs: int = 5, batch_size: int = 20):  # (d, n), (D, n)
    X = X.T  # (n, d)
    Z = Z.T  # (n, D)
    n, d = X.shape
    D = Z.shape[1]

    nn = TwoLayerRegressionNN(D, d_latent, d)  # represents mapping R^D -> R^d
    optimizer = torch.optim.Adam(nn.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(Z, X)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for idx, data in enumerate(dataloader):
            z_batch, x_batch = data  # (n_i, D), (n_i, d)
            z_batch = z_batch.T  # (D, n_i)
            x_batch = x_batch.T  # (d, n_i)

            optimizer.zero_grad()

            x_predicted = nn(z_batch)  # (d, n_i)
            loss = torch.linalg.norm(x_predicted - x_batch) ** 2 \
                   + lmbda * sum(torch.linalg.norm(Ai) ** 2 for Ai in nn.parameters())
            # i know you can use weight decay but wanted to be more explicit
            loss.backward()
            optimizer.step()
        X_predicted = nn(Z)
        print(f"Epoch {epoch}:")
        print(f"Squared error = ", torch.linalg.norm(X_predicted - X) ** 2)
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
    G_nn = train_twolayer_regression_nn(X, Z, d_latent, lmbda)
    return sum(torch.linalg.norm(Ai) ** 2 for Ai in G_nn.parameters())
