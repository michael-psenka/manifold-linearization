from itertools import chain
import pytorch_lightning as pl

from models.operators import *


class VAE(pl.LightningModule):
    def autoencode_data(self, X: torch.Tensor):
        raise NotImplementedError


class BetaVAE(VAE):
    def __init__(self, beta: float, d_x: int, d_z: int, d_latent: int, n_layers: int, lr: float):
        super(BetaVAE, self).__init__()
        self.beta: float = beta
        self.d_x: int = d_x
        self.d_z: int = d_z
        self.encoder: torch.nn.Module = fcnn(d_x, d_latent, d_latent, n_layers - 1)
        self.fc_mu: torch.nn.Module = torch.nn.Linear(d_latent, d_z)
        self.fc_var: torch.nn.Module = torch.nn.Linear(d_latent, d_z)
        self.decoder: torch.nn.Module = fcnn(d_z, d_x, d_latent, n_layers)
        self.lr: float = lr
        self.training_loss = []
        self.name = f"BetaVAE_beta{beta}_dx{d_x}_dz{d_z}_dl{d_latent}_l{n_layers}_lr{lr}"

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
        loss = self.beta * kl + recon_loss
        self.log_dict({"recon_loss": recon_loss, "kl": kl, "loss": loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def autoencode_data(self, X: torch.Tensor):
        return self.forward(X)


class FactorVAE(pl.LightningModule):
    def __init__(self, gamma: float, d_x: int, d_z: int, d_latent: int, n_layers: int, lr: float):
        super(FactorVAE, self).__init__()
        self.gamma: float = gamma
        self.d_x: int = d_x
        self.d_z: int = d_z
        self.encoder: torch.nn.Module = fcnn(d_x, d_latent, d_latent, n_layers - 1)
        self.fc_mu: torch.nn.Module = torch.nn.Linear(d_latent, d_z)
        self.fc_var: torch.nn.Module = torch.nn.Linear(d_latent, d_z)
        self.decoder: torch.nn.Module = fcnn(d_z, d_x, d_latent, n_layers)
        self.discriminator: torch.nn.Module = fcnn(d_z, 2, d_latent, n_layers)
        self.D_z_reserve = None
        self.lr: float = lr
        self.training_loss = []
        self.name = f"FactorVAE_gamma{gamma}_dx{d_x}_dz{d_z}_dl{d_latent}_l{n_layers}_lr{lr}"

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

    def permute_latent(self, z):
        B, D = z.size()

        # Returns a shuffled inds for each latent code in the batch
        inds = torch.cat([(D * i) + torch.randperm(D) for i in range(B)])
        return z.view(-1)[inds].view(B, D)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if isinstance(batch, tuple):
            x, _ = batch
        else:
            x = batch
        x_ll = self.encoder(x)
        mu = self.fc_mu(x_ll)
        log_var = self.fc_var(x_ll)
        p, q, z = self.sample(mu, log_var)
        x_hat = self.decoder(z)
        if optimizer_idx == 0:
            recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
            kl = torch.distributions.kl_divergence(q, p)
            kl = kl.mean()
            self.D_z_reserve = self.discriminator(z)
            vae_tc_loss = (self.D_z_reserve[:, 0] - self.D_z_reserve[:, 1]).mean()
            loss = kl + recon_loss + self.gamma * vae_tc_loss
            self.log_dict({"recon_loss": recon_loss, "kl": kl, "loss": loss})
        else:
            true_labels = torch.ones(x.size(0), dtype=torch.long,
                                     requires_grad=False)
            false_labels = torch.zeros(x.size(0), dtype=torch.long,
                                       requires_grad=False)

            z_perm = self.permute_latent(z)
            D_z_perm = self.discriminator(z_perm)
            loss = 0.5 * (torch.nn.functional.cross_entropy(self.D_z_reserve, false_labels) +
                          torch.nn.functional.cross_entropy(D_z_perm, true_labels))
            self.log_dict({"D_TC_loss": loss, "loss": loss})
        return loss

    def configure_optimizers(self):
        return [
                   torch.optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=self.lr),
                   torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
               ], []

    def autoencode_data(self, X: torch.Tensor):
        return self.forward(X)



def train_vanilla_vae(X: torch.Tensor, num_epochs: int = 100, batch_size: int = 32):
    trainer = pl.Trainer(max_epochs=num_epochs)
    dataloader = torch.utils.data.DataLoader(X, batch_size, drop_last=False)
    model = BetaVAE(beta=1.0, d_x=X.shape[1], d_z=X.shape[1], d_latent=X.shape[1], n_layers=5, lr=1e-3)
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


def train_beta_vae(X: torch.Tensor, num_epochs: int = 100, batch_size: int = 32):
    trainer = pl.Trainer(max_epochs=num_epochs)
    dataloader = torch.utils.data.DataLoader(X, batch_size, drop_last=False)
    model = BetaVAE(beta=4.0, d_x=X.shape[1], d_z=X.shape[1], d_latent=X.shape[1], n_layers=5, lr=1e-3)
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


def train_factor_vae(X: torch.Tensor, num_epochs: int = 100, batch_size: int = 32):
    trainer = pl.Trainer(max_epochs=num_epochs)
    dataloader = torch.utils.data.DataLoader(X, batch_size, drop_last=False)
    model = FactorVAE(gamma=30, d_x=X.shape[1], d_z=X.shape[1], d_latent=X.shape[1], n_layers=5, lr=1e-3)
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
