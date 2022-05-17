from collections import OrderedDict
from typing import Any, Dict, Sequence

import torch
import torch.distributions as dist


class LatentVariable(object):
    def __init__(self, name: str, kind: str, prob: str, dim: int, **kwargs: Any):
        self.name: str = name
        self.kind: str = kind  # "z", "c"
        self.dim: int = dim
        self.cdim: int = dim
        self.prob_name: str = prob  # "categorical", "normal", "uniform"

        # define probability distribution
        klass: Any = object
        if prob == "normal":
            klass = dist.normal.Normal(kwargs["mu"], kwargs["var"])
        elif prob == "uniform":
            klass = dist.uniform.Uniform(kwargs["min"], kwargs["max"])
        elif prob == "categorical":
            klass = Categorical(kwargs["k"])
            self.cdim = dim * kwargs["k"]

        self.prob: dist.Distribution = klass
        self.params = kwargs

    def __str__(self):
        return f"<LatentVariable(name: {self.name}, kind: {self.kind}, prob: {self.prob_name}, dim: {self.dim})>"

    def __repr__(self):
        return str(self)


def build_latent_variables(lv_configs: Dict) -> Dict[str, LatentVariable]:
    lvars: OrderedDict[str, LatentVariable] = OrderedDict()

    # first of all, add z variable
    count = 0
    for c in lv_configs:
        if c["kind"] == "z":
            if count > 1:
                raise Exception("More than two latent variables of kind 'z' exist!")
            lvars[c["name"]] = LatentVariable(**c)
            count += 1

    if count == 0:
        raise Exception("Latent variable of kind 'z' doesn't exist!")

    # after that, add other variables
    for c in lv_configs:
        if c["kind"] == "z":
            continue

        if c["name"] in lvars:
            raise Exception("Latent variable name is not unique.")

        lvars[c["name"]] = LatentVariable(**c)

    return lvars


class Categorical:
    def __init__(self, k: int):
        from torch.distributions.categorical import Categorical as _Categorical

        self.k = k

        p = torch.empty(k).fill_(1 / k)
        self.prob = _Categorical(p)

    def one_hot(self, x: torch.Tensor) -> torch.Tensor:
        b, c = tuple(x.shape)
        _x = torch.unsqueeze(x, 2)
        oh = torch.empty(b, c, self.k).zero_()
        oh.scatter_(2, _x, 1)
        oh = oh.view(b, c * self.k)

        return oh

    def sample(self, shape: Sequence[int]) -> torch.Tensor:
        x = self.prob.sample(shape)
        x = self.one_hot(x)

        return x