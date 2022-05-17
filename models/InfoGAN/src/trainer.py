import copy
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import loss
import util
from logger import Logger, MetricType
from variable import LatentVariable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Trainer(object):
    def __init__(
        self,
        dataloader: DataLoader,
        latent_vars: Dict[str, LatentVariable],
        models: Dict[str, nn.Module],
        optimizers: Dict[str, Any],
        losses: Dict[str, Any],
        configs: Dict[str, Any],
        logger: Logger,
    ):

        self.dataloader = dataloader
        self.latent_vars = latent_vars
        self.models = models
        self.optimizers = optimizers
        self.losses = losses
        self.configs = configs
        self.logger = logger

        self.device = util.current_device()

        self.grad_max_norm = configs["grad_max_norm"]

        self.n_log_samples = configs["n_log_samples"]

        self.gen_images_path = self.logger.path / "images"
        self.model_snapshots_path = self.logger.path / "models"
        for p in [self.gen_images_path, self.model_snapshots_path]:
            p.mkdir(parents=True, exist_ok=True)

        self.iteration = 0
        self.epoch = 0

        self.snapshot_models()

    def fix_seed(self):
        seed = self.configs["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def snapshot_models(self):
        for name, _model in self.models.items():
            model: nn.Module = copy.deepcopy(_model.module.cpu())
            torch.save(model, self.model_snapshots_path / f"{name}_model.pytorch")

    def snapshot_params(self):
        for name, model in self.models.items():
            torch.save(
                model.state_dict(),
                str(
                    self.model_snapshots_path
                    / f"{name}_params_{self.iteration:05d}.pytorch"
                ),
            )

    def log_random_images(self, n: int):
        img = util.gen_random_images(self.models["gen"], n)
        self.logger.tf_log_image(img, self.iteration, "random")
        save_image(img, self.gen_images_path / f"random_{self.iteration}.jpg")

    def log_images_discrete(self, var_name: str):
        img = util.gen_images_discrete(self.models["gen"], var_name)
        self.logger.tf_log_image(img, self.iteration, var_name)
        save_image(img, self.gen_images_path / f"{var_name}_{self.iteration}.jpg")

    def log_images_continuous(self, var_name: str, n: int):
        img = util.gen_images_continuous(self.models["gen"], var_name, n)
        self.logger.tf_log_image(img, self.iteration, f"{var_name}")
        save_image(img, self.gen_images_path / f"{var_name}_{self.iteration}.jpg")

    def train(self):
        # retrieve models and move them if necessary
        gen, dis = self.models["gen"], self.models["dis"]
        dhead, qhead = self.models["dhead"], self.models["qhead"]

        # move the model to appropriate device
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            gen, dis = nn.DataParallel(gen), nn.DataParallel(dis)
            dhead, qhead = nn.DataParallel(dhead), nn.DataParallel(qhead)

        gen, dis = gen.to(self.device), dis.to(self.device)
        dhead, qhead = dhead.to(self.device), qhead.to(self.device)

        # initialize model parameters
        weights_init(gen)
        weights_init(dis)
        weights_init(dhead)
        weights_init(qhead)

        # optimizers
        opt_gen = self.optimizers["gen"]
        opt_dis = self.optimizers["dis"]

        # losses
        adv_loss = self.losses["adv"]
        info_loss = self.losses["info"]

        # define metrics
        self.logger.define("iteration", MetricType.Number)
        self.logger.define("epoch", MetricType.Number)
        self.logger.define("loss_gen", MetricType.Loss)
        self.logger.define("loss_dis", MetricType.Loss)
        self.logger.define("loss_q", MetricType.Loss)

        for k, v in self.latent_vars.items():
            if v.kind == "z":
                continue
            self.logger.define(f"loss_{k}", MetricType.Loss)

        # start training
        self.logger.info(f"Start training, device: {self.device} n_gpus: {n_gpus}")
        self.logger.print_header()
        for i in range(self.configs["n_epochs"]):
            self.epoch += 1
            for x_real, _ in iter(self.dataloader):
                self.iteration += 1
                batchsize = len(x_real)

                gen.train()
                dis.train()
                dhead.train()
                qhead.train()

                # ------- discrminator phase -------
                opt_dis.zero_grad()

                # train with real
                x_real = x_real.to(self.device)
                y_real = dhead(dis(x_real))
                loss_dis_real = adv_loss(y_real, loss.LABEL_REAL)
                loss_dis_real.backward()

                # train with fake
                zs = gen.module.sample_latent_vars(batchsize)
                x_fake = gen.module.infer(list(zs.values()))
                y_fake = dhead(dis(x_fake.detach()))
                loss_dis_fake = adv_loss(y_fake, loss.LABEL_FAKE)

                loss_dis_fake.backward()
                clip_grad_norm(dis.parameters(), self.grad_max_norm)
                clip_grad_norm(dhead.parameters(), self.grad_max_norm)
                opt_dis.step()

                loss_dis = loss_dis_real + loss_dis_fake

                # ------- generator phase -------
                opt_gen.zero_grad()

                mid = dis(x_fake)
                y_fake, c_fake = dhead(mid), qhead(mid)

                # compute loss as fake samples are real
                loss_gen = adv_loss(y_fake, loss.LABEL_REAL)

                # compute mutual information loss
                c_true = {k: zs[k] for k in c_fake.keys()}
                loss_q, loss_q_details = info_loss(c_fake, c_true)
                loss_gen += loss_q

                loss_gen.backward()
                clip_grad_norm(gen.parameters(), self.grad_max_norm)
                clip_grad_norm(qhead.parameters(), self.grad_max_norm)
                opt_gen.step()

                # update metrics
                self.logger.update("iteration", self.iteration)
                self.logger.update("epoch", self.epoch)
                self.logger.update("loss_gen", loss_gen.cpu().item())
                self.logger.update("loss_dis", loss_dis.cpu().item())
                self.logger.update("loss_q", loss_q.cpu().item())

                for k, v in loss_q_details.items():
                    self.logger.update(f"loss_{k}", v.cpu().item())

                # log metrics
                if self.iteration % self.configs["log_interval"] == 0:
                    self.logger.log()
                    self.logger.log_tensorboard("iteration")
                    self.logger.clear()

                # snapshot models
                if self.iteration % self.configs["snapshot_interval"] == 0:
                    self.snapshot_params()

                # generate and save samples
                if self.iteration % self.configs["log_samples_interval"] == 0:
                    for var_name, var in self.latent_vars.items():
                        if var.kind == "z":
                            self.log_random_images(self.n_log_samples)
                        else:
                            if var.prob_name == "categorical":
                                self.log_images_discrete(var_name)
                            else:
                                self.log_images_continuous(var_name, self.n_log_samples)