import torch
from models.umap.umap import UMAP


def train_umap(X: torch.Tensor, dz: int = 0):
    u = UMAP(n_components=dz)
    X_np = X.detach().cpu().numpy()
    u.fit(X_np)

    def umap_encoding(x):
        x_np = x.detach().cpu().numpy()
        z = u.transform(x_np)
        return torch.from_numpy(z)

    def umap_decoding(z):
        z_np = z.detach().cpu().numpy()
        x = u.inverse_transform(z_np)
        return torch.from_numpy(x)

    return umap_encoding, umap_decoding
