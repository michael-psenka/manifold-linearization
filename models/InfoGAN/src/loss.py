import functools
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

import util
from variable import LatentVariable

LABEL_REAL, LABEL_FAKE = 1, 0


class AdversarialLoss:
    def __init__(self):
        self.loss = nn.BCELoss(reduction="mean")
        self.device = util.current_device()

    def __call__(self, y_hat: torch.Tensor, label: int):
        if label not in [LABEL_REAL, LABEL_FAKE]:
            raise Exception("Invalid label is passed to adversarial loss")

        y_true = torch.full(y_hat.size(), label, device=self.device)
        y_true = y_true.to(torch.float32)

        return self.loss(y_hat, y_true)


class InfoGANLoss:
    def __init__(self, latent_vars: Dict[str, LatentVariable]):
        self.latent_vars = latent_vars

        self.discrete_loss = nn.CrossEntropyLoss()
        self.continuous_loss = NormalNLLLoss()
        self.device = util.current_device()

    def __call__(
        self, cs_hat: Dict[str, torch.Tensor], cs_true: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if cs_hat.keys() != cs_true.keys():
            raise Exception("The keys of cs_hat is different from cs_true")

        losses: List[torch.Tensor] = []
        details: Dict[str, torch.Tensor] = {}
        for key in cs_hat.keys():
            c_hat, c_true = cs_hat[key], cs_true[key]

            if self.latent_vars[key].prob_name == "categorical":
                # loss for discrete variable
                _, targets = c_true.max(dim=1)
                loss = self.discrete_loss(c_hat, targets)
            elif self.latent_vars[key].prob_name == "normal":
                # loss for continuous variable
                dim: int = self.latent_vars[key].dim
                mean, ln_var = c_hat[:, :dim], c_hat[:, dim:]
                loss = self.continuous_loss(c_true, mean, ln_var)

            loss = loss * self.latent_vars[key].params["weight"]
            details[key] = loss
            losses.append(loss)

        return functools.reduce(lambda x, y: x + y, losses), details


class NormalNLLLoss:
    def __call__(
        self, x: torch.Tensor, mean: torch.Tensor, ln_var: torch.Tensor
    ) -> torch.Tensor:

        x_prec = torch.exp(-ln_var)
        x_diff = x - mean
        x_power = (x_diff * x_diff) * x_prec * -0.5
        loss = (ln_var + math.log(2 * math.pi)) / 2 - x_power

        return torch.mean(loss)