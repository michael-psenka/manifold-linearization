import argparse
import random
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import loss
import util
from logger import Logger
from model import DHead, Discriminator, Generator, QHead
from trainer import Trainer
from variable import build_latent_variables


def worker_init_fn(worker_id: int):
    random.seed(worker_id)


def create_optimizer(models: List[nn.Module], lr: float, decay: float):
    params: List[torch.Tensor] = []
    for m in models:
        params += list(m.parameters())
    return optim.Adam(params, lr=lr, betas=(0.5, 0.999), weight_decay=decay)


def main():
    configs = util.load_yaml('models/InfoGAN/config.yaml')

    dataset_name = 'mnist'
    dataset_path = './models/data'
    transform = transforms.Compose(
            [transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)])
    mnist = datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)
    idx = torch.tensor(mnist.targets) == 2
    dataset = torch.utils.data.dataset.Subset(mnist, np.where(idx==1)[0])
    dataloader = DataLoader(
        dataset,
        batch_size=2048,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    # prepare models
    latent_vars = build_latent_variables(configs["latent_variables"])
    gen, dis = Generator(latent_vars), Discriminator(configs["models"]["dis"])
    dhead, qhead = DHead(), QHead(latent_vars)
    models = {"gen": gen, "dis": dis, "dhead": dhead, "qhead": qhead}

    # prepare optimizers
    opt_gen = create_optimizer([gen, qhead], **configs["optimizer"]["gen"])
    opt_dis = create_optimizer([dis, dhead], **configs["optimizer"]["dis"])
    opts = {"gen": opt_gen, "dis": opt_dis}

    # prepare directories
    log_path = Path(configs["log_path"])
    log_path.mkdir(parents=True, exist_ok=True)
    tb_path = Path(configs["tensorboard_path"])
    tb_path.mkdir(parents=True, exist_ok=True)

    # initialize logger
    logger = Logger(log_path, tb_path)

    # initialize losses
    losses = {"adv": loss.AdversarialLoss(), "info": loss.InfoGANLoss(latent_vars)}

    # start training
    trainer = Trainer(
        dataloader, latent_vars, models, opts, losses, configs["training"], logger
    )
    trainer.train()


if __name__ == "__main__":
    main()