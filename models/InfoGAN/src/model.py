from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

import util
from variable import LatentVariable, build_latent_variables


class Noise(nn.Module):
    def __init__(self, use_noise: float, sigma: float = 0.2):
        super().__init__()
        self.use_noise = use_noise
        self.sigma = sigma
        self.device = util.current_device()

    def forward(self, x):
        if self.use_noise:
            return (
                x
                + self.sigma
                * torch.empty(
                    x.size(), device=self.device, requires_grad=False
                ).normal_()
            )
        return x


class Generator(nn.Module):
    def __init__(self, latent_vars: Dict[str, LatentVariable]):
        super().__init__()

        self.latent_vars = latent_vars
        self.dim_input = sum(map(lambda x: x.cdim, latent_vars.values()))
        ngf = 64
        self.device = util.current_device()

        # main layers
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.dim_input, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)

    def forward_dummy(self) -> torch.Tensor:
        shape = (2, self.dim_input, 1, 1)
        dummy_tensor = torch.empty(shape, device=self.device).normal_()
        return self.forward(dummy_tensor)

    def sample_latent_vars(self, batchsize: int) -> Dict[str, torch.Tensor]:
        zs: OrderedDict[str, torch.Tensor] = OrderedDict()

        for name, var in self.latent_vars.items():
            zs[name] = var.prob.sample([batchsize, var.dim])
            zs[name] = zs[name].to(self.device)

        return zs

    def infer(self, zs: List[torch.Tensor]) -> torch.Tensor:
        z = torch.cat(zs, dim=1)
        z = z.unsqueeze(-1).unsqueeze(-1)  # expand to image map

        x = self.forward(z)
        return x

    @property
    def module(self) -> nn.Module:
        return self


class Discriminator(nn.Module):
    def __init__(self, configs: Dict[str, Any]):
        super().__init__()

        self.configs = configs

        ndf = 64
        self.device = util.current_device()

        use_noise: bool = configs["use_noise"]
        noise_sigma: float = configs["noise_sigma"]

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )

    def forward(self, x):
        return self.main(x)

    def forward_dummy(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        shape = (2, 1, 64, 64)
        dummy_tensor = torch.empty(shape, device=self.device).normal_()

        return self.forward(dummy_tensor)

    @property
    def module(self) -> nn.Module:
        return self


class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 64
        self.main = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 3, 1, 1, bias=False),
            nn.Sigmoid(),
            # state size. 1 x 4 x 4
        )

    def forward(self, x):
        return self.main(x)

    @property
    def module(self) -> nn.Module:
        return self


class QHead(nn.Module):
    def __init__(self, latent_vars: Dict[str, LatentVariable]):
        super().__init__()

        ndf = 64
        self.main = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 2, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. 128 x 2 x 2
        )

        # generate each head module from latent variable
        self.convs = nn.ModuleDict()
        for name, var in latent_vars.items():
            if var.kind == "z":
                continue

            if var.prob_name == "normal":
                self.convs[name] = nn.Conv2d(ndf * 2, var.cdim * 2, 1)
            else:
                self.convs[name] = nn.Conv2d(ndf * 2, var.cdim, 1)

    def forward(self, x):
        mid = self.main(x)

        ys: Dict[str, torch.Tensor] = {}
        for name, conv in self.convs.items():
            ys[name] = conv(mid).squeeze()

        return ys

    @property
    def module(self) -> nn.Module:
        return self

