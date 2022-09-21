import pytorch_lightning as pl

from models.operators import *


class UnsupervisedVAE(pl.LightningModule):
    def autoencode_data(self, X: torch.Tensor):
        raise NotImplementedError


class SupervisedVAE(pl.LightningModule):
    def autoencode_data(self, X: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError


class UnsupervisedVanillaVAE(UnsupervisedVAE):
    def __init__(self, d_x: int, d_z: int, d_latent: int, n_layers: int, lr: float):
        super(UnsupervisedVanillaVAE, self).__init__()
        self.d_x: int = d_x
        self.d_z: int = d_z
        self.encoder: torch.nn.Module = fcnn(d_x, d_latent, d_latent, n_layers - 1)
        self.fc_mu: torch.nn.Module = torch.nn.Linear(d_latent, d_z)
        self.fc_var: torch.nn.Module = torch.nn.Linear(d_latent, d_z)
        self.decoder: torch.nn.Module = fcnn(d_z, d_x, d_latent, n_layers)
        self.lr: float = lr
        self.training_loss = []
        self.name = f"VanillaVAE_dx{d_x}_dz{d_z}_dl{d_latent}_l{n_layers}_lr{lr}"

    def forward(self, x):
        x_ll = self.encoder(x)
        mu = self.fc_mu(x_ll)
        log_var = self.fc_var(x_ll)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            x, _ = batch
        else:
            x = batch
        x_ll = self.encoder(x)
        mu = self.fc_mu(x_ll)
        log_var = self.fc_var(x_ll)
        p, q, z = self.sample(mu, log_var)
        x_hat = self.decoder(z)
        recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        loss = kl + recon_loss
        self.log_dict({"recon_loss": recon_loss, "kl": kl, "loss": loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def autoencode_data(self, X: torch.Tensor):
        return self.forward(X)


def train_vae(X: torch.Tensor, num_epochs: int = 100, batch_size: int = 32):
    trainer = pl.Trainer(max_epochs=num_epochs)
    dataloader = torch.utils.data.DataLoader(X, batch_size, drop_last=False)
    model = UnsupervisedVanillaVAE(d_x=X.shape[1], d_z=X.shape[1], d_latent=X.shape[1], n_layers=5, lr=1e-3)
    trainer.fit(model, train_dataloaders=dataloader)

    def vae_encoding(x):
        x_ll = model.encoder(x)
        mu = model.fc_mu(x_ll)
        log_var = model.fc_var(x_ll)
        p, q, z = model.sample(mu, log_var)
        return z

    def vae_decoding(z):
        return model.decoder(z)

    return vae_encoding, vae_decoding

