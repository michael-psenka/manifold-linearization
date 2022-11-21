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
        self.encoder: torch.nn.Module = mlp(d_x, d_latent, d_latent, n_layers - 1)
        self.fc_mu: torch.nn.Module = torch.nn.Linear(d_latent, d_z)
        self.fc_var: torch.nn.Module = torch.nn.Linear(d_latent, d_z)
        self.decoder: torch.nn.Module = mlp(d_z, d_x, d_latent, n_layers)
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
        self.encoder: torch.nn.Module = mlp(d_x, d_latent, d_latent, n_layers - 1)
        self.fc_mu: torch.nn.Module = torch.nn.Linear(d_latent, d_z)
        self.fc_var: torch.nn.Module = torch.nn.Linear(d_latent, d_z)
        self.decoder: torch.nn.Module = mlp(d_z, d_x, d_latent, n_layers)
        self.discriminator: torch.nn.Module = mlp(d_z, 2, d_latent, n_layers)
        self.lr: float = lr
        self.training_loss = []
        self.name = f"FactorVAE_gamma{gamma}_dx{d_x}_dz{d_z}_dl{d_latent}_l{n_layers}_lr{lr}"
        self.automatic_optimization = False

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

    def training_step(self, batch, batch_idx):
        opt_vae, opt_d = self.optimizers()

        if isinstance(batch, tuple):
            x, _ = batch
        else:
            x = batch

        bs = x.shape[0]
        x1, x2 = x[:bs//2].clone(), x[bs//2:].clone()

        x1_ll = self.encoder(x1)
        mu = self.fc_mu(x1_ll)
        log_var = self.fc_var(x1_ll)
        p, q, z1 = self.sample(mu, log_var)
        x1_hat = self.decoder(z1)

        recon_loss = torch.nn.functional.mse_loss(x1_hat, x1, reduction="mean")
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        D_z1 = self.discriminator(z1)

        vae_tc_loss = (D_z1[:, 0] - D_z1[:, 1]).mean()
        vae_loss = kl + recon_loss + self.gamma * vae_tc_loss

        opt_vae.zero_grad()
        self.manual_backward(vae_loss, retain_graph=True)

        zeros = torch.zeros(x1.size(0), dtype=torch.long, requires_grad=False)
        ce_orig = torch.nn.functional.cross_entropy(D_z1, zeros)

        x2_ll = self.encoder(x2)
        mu = self.fc_mu(x2_ll)
        log_var = self.fc_var(x2_ll)
        p, q, z2 = self.sample(mu, log_var)

        z2_perm = self.permute_latent(z2)
        D_z2_perm = self.discriminator(z2_perm)

        ones = torch.ones(x2.size(0), dtype=torch.long, requires_grad=False)
        ce_perm = torch.nn.functional.cross_entropy(D_z2_perm, ones)
        d_loss = 0.5 * (ce_orig + ce_perm)

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_vae.step()
        opt_d.step()

    def configure_optimizers(self):
        return [
                   torch.optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=self.lr),
                   torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
               ], []

    def autoencode_data(self, X: torch.Tensor):
        return self.forward(X)



def train_vanilla_vae(X: torch.Tensor, d_z: int = 0, d_latent: int = 0, n_layers: int = 5,
                      lr: float = 1e-3, num_epochs: int = 100, batch_size: int = 32):
    d_z = d_z or X.shape[1]
    d_latent = d_latent or d_z
    trainer = pl.Trainer(max_epochs=num_epochs)
    dataloader = torch.utils.data.DataLoader(X, batch_size, drop_last=False)
    model = BetaVAE(beta=1.0, d_x=X.shape[1], d_z=d_z, d_latent=d_latent, n_layers=n_layers, lr=lr)
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


def train_beta_vae(X: torch.Tensor, beta: float = 4.0, d_z: int = 0, d_latent: int = 0, n_layers: int = 5,
                   lr: float = 1e-3, num_epochs: int = 100, batch_size: int = 32):
    d_z = d_z or X.shape[1]
    d_latent = d_latent or d_z
    trainer = pl.Trainer(max_epochs=num_epochs)
    dataloader = torch.utils.data.DataLoader(X, batch_size, drop_last=False)
    model = BetaVAE(beta=beta, d_x=X.shape[1], d_z=d_z, d_latent=d_latent, n_layers=n_layers, lr=lr)
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


def train_factor_vae(X: torch.Tensor, gamma: float = 30.0, d_z: int = 0, d_latent: int = 0, n_layers: int = 5,
                   lr: float = 1e-3, num_epochs: int = 100, batch_size: int = 32):
    d_z = d_z or X.shape[1]
    d_latent = d_latent or d_z
    trainer = pl.Trainer(max_epochs=num_epochs)
    dataloader = torch.utils.data.DataLoader(X, batch_size, drop_last=False)
    model = FactorVAE(gamma=gamma, d_x=X.shape[1], d_z=d_z, d_latent=d_latent, n_layers=n_layers, lr=lr)
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
