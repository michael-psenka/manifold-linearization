from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch import nn
from torchvision.utils import make_grid


def load_yaml(path: str) -> Dict[str, Any]:
    f = open(path)
    c = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    return c


def current_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def gen_random_images(gen: nn.Module, n: int) -> np.ndarray:
    """
    Generate a n*n grid image with random samples
    """
    gen.eval()
    with torch.no_grad():
        zs = gen.module.sample_latent_vars(n * n)
        x = gen.module.infer(list(zs.values()))

    x = make_grid(x, n, normalize=True)

    return x


def gen_images_discrete(gen: nn.Module, var_name: str):
    """
    Generate a k*k grid image with varying discrete variable
    """

    k: int = gen.latent_vars[var_name].params["k"]
    gen.eval()
    with torch.no_grad():
        zs = gen.module.sample_latent_vars(k * k)

        # overwrite variable(e.g. c1) to intentional values
        idx = np.arange(k).repeat(k)
        one_hot = np.zeros((k * k, k))
        one_hot[range(k * k), idx] = 1
        zs[var_name] = torch.tensor(one_hot, device=current_device(), dtype=torch.float)

        x = gen.module.infer(list(zs.values()))

    x = make_grid(x, k, normalize=True)

    return x


def gen_images_continuous(gen: nn.Module, var_name: str, n: int):
    """
    Generate a n*n grid image with varying continuous variable
    """

    _min: int = -2
    _max: int = 2

    gen.eval()
    with torch.no_grad():
        zs = gen.module.sample_latent_vars(n)

        for _var_name, var in gen.latent_vars.items():
            if _var_name == var_name:
                # overwrite continuous variable to intentional values
                interp = np.linspace(_min, _max, n)
                interp = np.expand_dims(interp, 1)
                interp = np.tile(interp, (n, 1))
                zs[_var_name] = torch.tensor(
                    interp, device=current_device(), dtype=torch.float
                )
            else:
                # replicate n times.
                # (n, c) -> (n, 1, c) -> (n, n, c) -> (n*n, c)
                zs[_var_name] = (
                    zs[_var_name].view(n, 1, -1).repeat(1, n, 1).view(n * n, -1)
                )
        x = gen.module.infer(list(zs.values()))

    x = make_grid(x, n, normalize=True)

    return x