import itertools

import pytorch_lightning as pl

from models.operators import *


class UnsupervisedGAN(pl.LightningModule):
    def generate_data(self, n: int):
        raise NotImplementedError


class SupervisedGAN(pl.LightningModule):
    def generate_data(self, n: int, y: torch.Tensor):
        raise NotImplementedError


class UnsupervisedVanillaGAN(UnsupervisedGAN):
    def __init__(self, d_x: int, d_noise: int, d_latent: int, n_layers: int, lr: float):
        super(UnsupervisedVanillaGAN, self).__init__()
        self.d_x: int = d_x
        self.d_noise: int = d_noise
        self.discriminator: torch.nn.Module = torch.nn.Sequential(fcnn(d_x, 1, d_latent, n_layers), torch.nn.Sigmoid())
        self.generator: torch.nn.Module = fcnn(d_noise, d_x, d_latent, n_layers)
        self.lr: float = lr
        self.training_loss = []
        self.name = f"VanillaGAN_dx{d_x}_dn{d_noise}_dl{d_latent}_l{n_layers}_lr{lr}"

    def forward(self, z):
        return self.generator(z)

    def generator_loss(self, x):
        z = torch.randn(x.shape[0], self.d_noise, device=self.device)
        y = torch.ones(x.shape[0], 1, device=self.device)
        gen = self.generator(z)
        D_output = self.discriminator(gen)
        g_loss = torch.nn.functional.binary_cross_entropy(D_output, y)
        return g_loss

    def discriminator_loss(self, x):
        b = x.shape[0]
        x_real = x
        y_real = torch.zeros(b, 1, device=self.device)
        D_output = self.discriminator(x_real)
        D_real_loss = torch.nn.functional.binary_cross_entropy(D_output, y_real)
        z = torch.randn(b, self.d_noise, device=self.device)
        x_fake = self(z)
        y_fake = torch.ones(b, 1, device=self.device)
        D_output = self.discriminator(x_fake)
        D_fake_loss = torch.nn.functional.binary_cross_entropy(D_output, y_fake)
        D_loss = D_real_loss + D_fake_loss
        return D_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        if isinstance(batch, tuple):
            x, _ = batch
        else:
            x = batch
        result = None
        if optimizer_idx == 0:
            result = self.generator_step(x)
        if optimizer_idx == 1:
            result = self.discriminator_step(x)
        return result

    def generator_step(self, x):
        g_loss = self.generator_loss(x)
        self.log("g_loss", g_loss, on_epoch=True, prog_bar=True)
        return g_loss

    def discriminator_step(self, x):
        d_loss = self.discriminator_loss(x)
        self.log("d_loss", d_loss, on_epoch=True, prog_bar=True)
        return d_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return [opt_g, opt_d], []

    def generate_data(self, n: int):
        Z = torch.randn(size=(n, self.d_noise))
        gZ = self.generator(Z)
        return gZ


class UnsupervisedInfoGAN(SupervisedGAN):
    def __init__(self, d_x: int, d_noise: int, d_code: int, d_latent: int, n_layers: int, lr: float):
        super(UnsupervisedInfoGAN, self).__init__()
        self.d_x: int = d_x
        self.d_noise: int = d_noise
        self.d_code: int = d_code
        self.encoder: torch.nn.Module = fcnn(d_x, d_noise, d_latent, n_layers - 1)
        self.discriminator_layer: torch.nn.Module = torch.nn.Sequential(torch.nn.Linear(d_latent, 1),
                                                                        torch.nn.Sigmoid())
        self.code_layer: torch.nn.Module = torch.nn.Linear(d_latent, d_code)
        self.generator: torch.nn.Module = fcnn(d_noise + d_code, d_x, d_latent, n_layers)
        self.lr: float = lr
        self.training_loss = []
        self.name = f"InfoGAN_dx{d_x}_dn{d_noise}_dc{d_code}_dl{d_latent}_l{n_layers}_lr{lr}"

    def forward(self, z, c):
        return self.generator(torch.cat((z, c), -1))

    def generator_loss(self, x):
        n = x.shape[0]
        fake = torch.ones(n, 1, device=self.device)
        z = torch.randn(n, self.d_noise, device=self.device)
        c = 2 * torch.rand(n, self.d_code, device=self.device) - 1
        x_gen = self(z, c)
        discrim = self.discriminator_layer(self.encoder(x_gen))
        g_loss = torch.nn.functional.binary_cross_entropy(discrim, fake)
        return g_loss

    def discriminator_loss(self, x):
        n = x.shape[0]
        real = torch.zeros(n, 1, device=self.device)
        fake = torch.ones(n, 1, device=self.device)
        x_real = x
        z = torch.randn(n, self.d_noise, device=self.device)
        c = 2 * torch.rand(n, self.d_code, device=self.device) - 1
        x_gen = self(z, c)
        discrim_real = self.discriminator_layer(self.encoder(x_real))
        discrim_fake = self.discriminator_layer(self.encoder(x_gen))
        d_loss = (torch.nn.functional.binary_cross_entropy(discrim_real,
                                                           real) + torch.nn.functional.binary_cross_entropy(
            discrim_fake, fake)) / 2.0
        return d_loss

    def info_loss(self, x):
        n = x.shape[0]
        z = torch.randn(size=(n, self.d_noise))
        c = 2 * torch.rand(size=(n, self.d_code)) - 1
        x_gen = self(z, c)
        x_enc = self.encoder(x_gen)
        enc_code = self.code_layer(x_enc)
        info_loss = torch.nn.functional.mse_loss(enc_code, c)
        return info_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch
        result = None
        if optimizer_idx == 0:
            result = self.generator_step(x)
        if optimizer_idx == 1:
            result = self.discriminator_step(x)
        if optimizer_idx == 2:
            result = self.info_step(x)
        return result

    def generator_step(self, x):
        g_loss = self.generator_loss(x)
        self.log("g_loss", g_loss, on_epoch=True, prog_bar=True)
        return g_loss

    def discriminator_step(self, x):
        d_loss = self.discriminator_loss(x)
        self.log("d_loss", d_loss, on_epoch=True, prog_bar=True)
        return d_loss

    def info_step(self, x):
        i_loss = self.info_loss(x)
        self.log("i_loss", i_loss, on_epoch=True, prog_bar=True)
        return i_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(itertools.chain(
            self.encoder.parameters(), self.discriminator_layer.parameters(), self.classification_layer.parameters(),
            self.code_layer.parameters()
        ), lr=self.lr)
        opt_i = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [opt_g, opt_d, opt_i], []

    def generate_data(self, n: int, y: torch.Tensor):
        Z = torch.randn(size=(n, self.d_noise))
        c = torch.rand(size=(n, self.d_code))
        gZ = self(Z, y, c)
        return gZ


def train_infogan(X: torch.Tensor, num_epochs: int = 100, batch_size: int = 32):
    trainer = pl.Trainer(max_epochs=num_epochs)
    dataloader = torch.utils.data.DataLoader(X, batch_size, drop_last=False)
    model = UnsupervisedInfoGAN(k=1, d_x=X.shape[1], d_noise=X.shape[1], d_code=5, d_latent=X.shape[1], n_layers=5, lr=1e-3)
    trainer.fit(model, train_dataloaders=dataloader)

    def infogan_encoding(x):
        return model.encoder(x)

    def infogan_decoding(z):
        n = z.shape[0]
        c = 2 * torch.rand(size=(n, model.d_code)) - 1
        return model(z, c)

    return infogan_encoding, infogan_decoding
