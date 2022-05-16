import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mimicry as mmc
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets

import torchvision.transforms as transforms

from torch_mimicry.nets.dcgan import dcgan_base
from torch_mimicry.modules.layers import SNLinear
from torch_mimicry.nets import sngan
from torch_mimicry.nets.sngan.sngan_128 import SNGANDiscriminator128
from torch_mimicry.nets.sngan.sngan_48 import SNGANDiscriminator48
from torch_mimicry.nets.sngan.sngan_32 import SNGANDiscriminator32
from torch_mimicry.training import scheduler, logger, metric_log

import time
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms.functional as FF
import torchvision.utils as vutils

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

class Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x)


class customSNGANDiscriminator128(SNGANDiscriminator128):

    def __init__(self, nz=128, ndf=1024, **kwargs):
        super(customSNGANDiscriminator128, self).__init__(ndf, **kwargs)
        self.nz = nz
        self.l7 = nn.Sequential(SNLinear(self.ndf, nz), Norm())


class customSNGANDiscriminator48(SNGANDiscriminator48):

    def __init__(self, nz=128, ndf=1024, **kwargs):
        super(customSNGANDiscriminator48, self).__init__(ndf, **kwargs)
        self.nz = nz
        self.l5 = nn.Sequential(SNLinear(self.ndf, nz), Norm())


class customSNGANDiscriminator32(SNGANDiscriminator32):

    def __init__(self, nz=128, ndf=128, **kwargs):
        super(customSNGANDiscriminator32, self).__init__(ndf, **kwargs)
        self.nz = nz
        self.l5 = nn.Sequential(SNLinear(self.ndf, nz), Norm())

class GeneratorMNIST(dcgan_base.DCGANBaseGenerator):
    r"""
    ResNet backbone generator for ResNet DCGAN.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz=128, ngf=64, bottom_width=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        # self.main = nn.Sequential(
        #     nn.Linear(nz, ngf * 8),
        #     nn.BatchNorm1d(ngf * 8),
        #     nn.ReLU(True),
        #     nn.Linear(ngf * 8, 7*7*128),
        #     nn.BatchNorm1d(7*7*128),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1, bias=False),
        #     nn.Tanh()
        # )
        
        self.main = nn.Sequential(
            nn.Linear(nz, ngf * 16),
            nn.BatchNorm1d(ngf * 16),
            nn.ReLU(True),
            nn.Linear(ngf * 16, 7*7*ngf * 2),
            nn.BatchNorm1d(7*7*ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.
        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).
        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        # h = h.view(x.shape[0], -1, 1, 1)
        return self.main(x.view(x.shape[0], -1, 1, 1))


class DiscriminatorMNIST(dcgan_base.DCGANBaseDiscriminator):
    r"""
    ResNet backbone discriminator for ResNet DCGAN.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz=128, ndf=64, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        self.nz = nz
        # self.main = nn.Sequential(
        #     nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(ndf * 4, nz, 4, 1, 0, bias=False),
        #     nn.Flatten()  # new
        #     # nn.LeakyReLU(0.2, inplace=True), #New
        #     # nn.Linear(ndf, ndf, bias=False)
        #     # nn.Sigmoid()
        # )
        self.main = nn.Sequential(# 28 -> 14
            nn.Conv2d(1, ndf, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1),
            nn.Conv2d(ndf, 2*ndf, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2*ndf),
            nn.LeakyReLU(0.1),
            nn.Linear(2*ndf*7*7, 16*ndf),
            nn.BatchNorm1d(16*ndf),
            nn.LeakyReLU(0.1),
            nn.Linear(16*ndf, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.
        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).
        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        return F.normalize(self.main(x))

def weights_init_mnist_model(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class MCRGANloss(nn.Module):

    def __init__(self, gam1=1., gam2=1., gam3=1., eps=0.5, numclasses=1000, mode=1, rho=None):
        super(MCRGANloss, self).__init__()

        self.num_class = numclasses
        self.train_mode = mode
        self.faster_logdet = False
        self.gam1 = gam1
        self.gam2 = gam2
        self.gam3 = gam3
        self.eps = eps

    def forward(self, Z, Z_bar, real_label, ith_inner_loop, num_inner_loop):

        # t = time.time()
        # errD, empi = self.old_version(Z, Z_bar, real_label, ith_inner_loop, num_inner_loop)
        errD, empi = self.fast_version(Z, Z_bar, real_label, ith_inner_loop, num_inner_loop)
        # print("faster version time: ", time.time() - t)
        # print("faster errD", errD)

        return errD, empi

    def old_version(self, Z, Z_bar, real_label, ith_inner_loop, num_inner_loop):

        """ original version, need to calculate 52 times log-det"""
        if self.train_mode == 2:
            loss_z, _ = self.deltaR(Z, real_label, self.num_class)
            assert num_inner_loop >= 2
            if (ith_inner_loop + 1) % num_inner_loop != 0:
                return loss_z, None

            loss_h, _ = self.deltaR(Z_bar, real_label, self.num_class)
            errD = self.gam1 * loss_z + self.gam2 * loss_h
            empi = [loss_z, loss_h]
            term3 = 0.

            for i in range(self.num_class):
                new_Z = torch.cat((Z[real_label == i], Z_bar[real_label == i]), 0)
                new_label = torch.cat(
                    (torch.zeros_like(real_label[real_label == i]),
                     torch.ones_like(real_label[real_label == i]))
                )
                loss, _ = self.deltaR(new_Z, new_label, 2)
                term3 += loss
            empi = empi + [term3]
            errD += self.gam3 * term3

        elif self.train_mode == 1:

            loss_z, _ = self.deltaR(Z, real_label, self.num_class)
            loss_h, _ = self.deltaR(Z_bar, real_label, self.num_class)
            errD = self.gam1 * loss_z + self.gam2 * loss_h
            empi = [loss_z, loss_h]
            term3 = 0.

            for i in range(self.num_class):
                new_Z = torch.cat((Z[real_label == i], Z_bar[real_label == i]), 0)
                new_label = torch.cat(
                    (torch.zeros_like(real_label[real_label == i]),
                     torch.ones_like(real_label[real_label == i]))
                )
                loss, _ = self.deltaR(new_Z, new_label, 2)
                term3 += loss
            empi = empi + [term3]
            errD += self.gam3 * term3
        elif self.train_mode == 0:
            new_Z = torch.cat((Z, Z_bar), 0)
            new_label = torch.cat((torch.zeros_like(real_label), torch.ones_like(real_label)))
            errD, em = self.deltaR(new_Z, new_label, 2)
            empi = (em[0], em[1])
        else:
            raise ValueError()

        return errD, empi

    def fast_version(self, Z, Z_bar, real_label, ith_inner_loop, num_inner_loop):

        """ decrease the times of calculate log-det  from 52 to 32"""

        if self.train_mode == 2:
            z_total, (z_discrimn_item, z_compress_item, z_compress_losses, z_scalars) = self.deltaR(Z, real_label,
                                                                                                    self.num_class)
            assert num_inner_loop >= 2
            if (ith_inner_loop + 1) % num_inner_loop != 0:
                # print(f"{ith_inner_loop + 1}/{num_inner_loop}")
                # print("calculate delta R(z)")
                return z_total, None

            zbar_total, (zbar_discrimn_item, zbar_compress_item, zbar_compress_losses, zbar_scalars) = self.deltaR(
                Z_bar, real_label, self.num_class)
            empi = [z_total, zbar_total]

            itemRzjzjbar = 0.
            for j in range(self.num_class):
                new_z = torch.cat((Z[real_label == j], Z_bar[real_label == j]), 0)
                R_zjzjbar = self.compute_discrimn_loss(new_z.T)
                itemRzjzjbar += R_zjzjbar

            errD_ = self.gam1 * (z_discrimn_item - z_compress_item) + \
                    self.gam2 * (zbar_discrimn_item - zbar_compress_item) + \
                    self.gam3 * (itemRzjzjbar - 0.25 * sum(z_compress_losses) - 0.25 * sum(zbar_compress_losses))
            errD = -errD_

            empi = empi + [-itemRzjzjbar + 0.25 * sum(z_compress_losses) + 0.25 * sum(zbar_compress_losses)]
            # print("calculate multi")

        elif self.train_mode == 1:
            z_total, (z_discrimn_item, z_compress_item, z_compress_losses, z_scalars) = self.deltaR(Z, real_label, self.num_class)
            zbar_total, (zbar_discrimn_item, zbar_compress_item, zbar_compress_losses, zbar_scalars) = self.deltaR(Z_bar, real_label, self.num_class)
            empi = [z_total, zbar_total]

            itemRzjzjbar = 0.
            for j in range(self.num_class):
                new_z = torch.cat((Z[real_label == j], Z_bar[real_label == j]), 0)
                R_zjzjbar = self.compute_discrimn_loss(new_z.T)
                itemRzjzjbar += R_zjzjbar

            errD_ = self.gam1 * (z_discrimn_item - z_compress_item) + \
                    self.gam2 * (zbar_discrimn_item - zbar_compress_item) + \
                    self.gam3 * (itemRzjzjbar - 0.25 * sum(z_compress_losses) - 0.25 * sum(zbar_compress_losses))
            errD = -errD_

            empi = empi + [-itemRzjzjbar + 0.25 * sum(z_compress_losses) + 0.25 * sum(zbar_compress_losses)]

        elif self.train_mode == 0:
            new_Z = torch.cat((Z, Z_bar), 0)
            new_label = torch.cat((torch.zeros_like(real_label), torch.ones_like(real_label)))
            errD, extra = self.deltaR(new_Z, new_label, 2)
            empi = (extra[0], extra[1])

        elif self.train_mode == 10:
            errD, empi = self.double_loop(Z, Z_bar, real_label, ith_inner_loop, num_inner_loop)
        else:
            raise ValueError()

        return errD, empi

    def logdet(self, X):

        if self.faster_logdet:
            return 2 * torch.sum(torch.log(torch.diag(torch.linalg.cholesky(X, upper=True))))
        else:
            return torch.logdet(X)

    def compute_discrimn_loss(self, Z):
        """Theoretical Discriminative Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        scalar = d / (n * self.eps)
        logdet = self.logdet(I + scalar * Z @ Z.T)
        return logdet / 2.

    def compute_compress_loss(self, Z, Pi):
        """Theoretical Compressive Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        compress_loss = []
        scalars = []
        for j in range(Pi.shape[1]):
            Z_ = Z[:, Pi[:, j] == 1]
            trPi = Pi[:, j].sum() + 1e-8
            scalar = d / (trPi * self.eps)
            log_det = 1. if Pi[:, j].sum() == 0 else self.logdet(I + scalar * Z_ @ Z_.T)
            compress_loss.append(log_det)
            scalars.append(trPi / (2 * n))
        return compress_loss, scalars

    def deltaR(self, Z, Y, num_classes):

        Pi = F.one_hot(Y, num_classes).to(Z.device)
        discrimn_loss = self.compute_discrimn_loss(Z.T)
        compress_loss, scalars = self.compute_compress_loss(Z.T, Pi)

        compress_term = 0.
        for z, s in zip(compress_loss, scalars):
            compress_term += s * z
        total_loss = discrimn_loss - compress_term

        return -total_loss, (discrimn_loss, compress_term, compress_loss, scalars)

class MUltiGPUTrainer(mmc.training.Trainer):

    def __init__(self,
                 netD,
                 netG,
                 optD,
                 optG,
                 dataloader,
                 num_steps,
                 log_dir='./logs',
                 n_dis=1,
                 lr_decay=None,
                 device=None,
                 netG_ckpt_file=None,
                 netD_ckpt_file=None,
                 print_steps=1,
                 vis_steps=500,
                 log_steps=50,
                 save_steps=5000,
                 flush_secs=30,
                 amp=False):

        # Input values checks
        ints_to_check = {
            'num_steps': num_steps,
            'n_dis': n_dis,
            'print_steps': print_steps,
            'vis_steps': vis_steps,
            'log_steps': log_steps,
            'save_steps': save_steps,
            'flush_secs': flush_secs
        }
        for name, var in ints_to_check.items():
            if var < 1:
                raise ValueError('{} must be at least 1 but got {}.'.format(
                    name, var))

        self.netD = netD
        self.netG = netG
        self.optD = optD
        self.optG = optG
        self.n_dis = n_dis
        self.lr_decay = lr_decay
        self.dataloader = dataloader
        self.num_steps = num_steps
        self.device = device
        self.log_dir = log_dir
        self.netG_ckpt_file = netG_ckpt_file
        self.netD_ckpt_file = netD_ckpt_file
        self.print_steps = print_steps
        self.vis_steps = vis_steps
        self.log_steps = log_steps
        self.save_steps = save_steps
        self.amp = amp
        self.parallel = isinstance(self.netG, nn.DataParallel)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Training helper objects
        self.logger = logger.Logger(log_dir=self.log_dir,
                                    num_steps=self.num_steps,
                                    dataset_size=len(self.dataloader),
                                    flush_secs=flush_secs,
                                    device=self.device)

        self.scheduler = scheduler.LRScheduler(lr_decay=self.lr_decay,
                                               optD=self.optD,
                                               optG=self.optG,
                                               num_steps=self.num_steps)

        # Obtain custom or latest checkpoint files
        if self.netG_ckpt_file:
            self.netG_ckpt_dir = os.path.dirname(netG_ckpt_file)
            self.netG_ckpt_file = netG_ckpt_file
        else:
            self.netG_ckpt_dir = os.path.join(self.log_dir, 'checkpoints',
                                              'netG')
            self.netG_ckpt_file = self._get_latest_checkpoint(
                self.netG_ckpt_dir)  # can be None

        if self.netD_ckpt_file:
            self.netD_ckpt_dir = os.path.dirname(netD_ckpt_file)
            self.netD_ckpt_file = netD_ckpt_file
        else:
            self.netD_ckpt_dir = os.path.join(self.log_dir, 'checkpoints',
                                              'netD')
            self.netD_ckpt_file = self._get_latest_checkpoint(
                self.netD_ckpt_dir)

        # Log hyperparameters for experiments
        self.params = {
            'log_dir': self.log_dir,
            'num_steps': self.num_steps,
            'batch_size': self.dataloader.batch_size,
            'n_dis': self.n_dis,
            'lr_decay': self.lr_decay,
            'optD': optD.__repr__(),
            'optG': optG.__repr__(),
            'save_steps': self.save_steps,
        }
        self._log_params(self.params)

        # Device for hosting model and data
        if not self.device:
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else "cpu")

    def _save_model_checkpoints(self, global_step):
        """
        Saves both discriminator and generator checkpoints.
        """
        if not self.parallel:
            self.netG.save_checkpoint(directory=self.netG_ckpt_dir,
                                      global_step=global_step,
                                      optimizer=self.optG)

            self.netD.save_checkpoint(directory=self.netD_ckpt_dir,
                                      global_step=global_step,
                                      optimizer=self.optD)
        else:
            self.netG.module.save_checkpoint(directory=self.netG_ckpt_dir,
                                             global_step=global_step,
                                             optimizer=self.optG)

            self.netD.module.save_checkpoint(directory=self.netD_ckpt_dir,
                                             global_step=global_step,
                                             optimizer=self.optD)

    def _restore_models_and_step(self):
        """
        Restores model and optimizer checkpoints and ensures global step is in sync.
        """
        global_step_D = global_step_G = 0

        if self.netD_ckpt_file and os.path.exists(self.netD_ckpt_file):
            print("INFO: Restoring checkpoint for D...")
            global_step_D = self.netD.module.restore_checkpoint(
                ckpt_file=self.netD_ckpt_file, optimizer=self.optD)

        if self.netG_ckpt_file and os.path.exists(self.netG_ckpt_file):
            print("INFO: Restoring checkpoint for G...")
            global_step_G = self.netG.module.restore_checkpoint(
                ckpt_file=self.netG_ckpt_file, optimizer=self.optG)

        if global_step_G != global_step_D:
            raise ValueError('G and D Networks are out of sync.')
        else:
            global_step = global_step_G  # Restores global step

        return global_step

    def train(self):
        """
        Runs the training pipeline with all given parameters in Trainer.
        """
        # Restore models
        global_step = self._restore_models_and_step()
        print("INFO: Starting training from global step {}...".format(
            global_step))

        try:
            start_time = time.time()

            # Iterate through data
            iter_dataloader = iter(self.dataloader)
            while global_step < self.num_steps:
                log_data = metric_log.MetricLog()  # log data for tensorboard

                # -------------------------
                #   One Training Step
                # -------------------------
                # Update n_dis times for D
                for i in range(self.n_dis):
                    iter_dataloader, real_batch = self._fetch_data(
                        iter_dataloader=iter_dataloader)

                    # ------------------------
                    #   Update D Network
                    # -----------------------

                    self.netD.module.zero_grad()
                    real_images, real_labels = real_batch
                    batch_size = real_images.shape[0]  # Match batch sizes for last iter

                    # Produce logits for real images
                    output_real = self.netD(real_images)

                    # Produce fake images
                    noise = torch.randn((batch_size, self.netG.module.nz), device=self.device)
                    fake_images = self.netG(noise).detach()

                    # Produce logits for fake images
                    output_fake = self.netD(fake_images)

                    # Compute loss for D
                    errD = self.netD.module.compute_gan_loss(output_real=output_real,
                                                             output_fake=output_fake)

                    # Backprop and update gradients
                    errD.backward()
                    self.optD.step()

                    # Compute probabilities
                    D_x, D_Gz = self.netD.module.compute_probs(output_real=output_real,
                                                               output_fake=output_fake)

                    # Log statistics for D once out of loop
                    log_data.add_metric('errD', errD.item(), group='loss')
                    log_data.add_metric('D(x)', D_x, group='prob')
                    log_data.add_metric('D(G(z))', D_Gz, group='prob')

                    # -----------------------
                    #   Update G Network
                    # -----------------------
                    # Update G, but only once.
                    if i == (self.n_dis - 1):

                        self.netG.module.zero_grad()

                        # Get only batch size from real batch
                        batch_size = real_batch[0].shape[0]

                        # Produce fake images
                        noise = torch.randn((batch_size, self.netG.module.nz), device=self.device)
                        fake_images = self.netG(noise)

                        # Compute output logit of D thinking image real
                        output = self.netD(fake_images)

                        # Compute loss
                        errG = self.netG.module.compute_gan_loss(output=output)

                        # Backprop and update gradients
                        errG.backward()
                        self.optG.step()

                        # Log statistics
                        log_data.add_metric('errG', errG, group='loss')

                # --------------------------------
                #   Update Training Variables
                # -------------------------------
                log_data = self.scheduler.step(log_data=log_data,
                                               global_step=global_step)

                # -------------------------
                #   Logging and Metrics
                # -------------------------
                if global_step % self.log_steps == 0:
                    self.logger.write_summaries(log_data=log_data,
                                                global_step=global_step)

                if global_step % self.print_steps == 0:
                    curr_time = time.time()
                    self.logger.print_log(global_step=global_step,
                                          log_data=log_data,
                                          time_taken=(curr_time - start_time) /
                                          self.print_steps)
                    start_time = curr_time

                if global_step % self.vis_steps == 0:
                    self.logger.vis_images(netG=self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG,
                                           global_step=global_step)

                if global_step % self.save_steps == 0:
                    print("INFO: Saving checkpoints...")
                    self._save_model_checkpoints(global_step)

                global_step += 1

            print("INFO: Saving final checkpoints...")
            self._save_model_checkpoints(global_step)

        except KeyboardInterrupt:
            print("INFO: Saving checkpoints from keyboard interrupt...")
            self._save_model_checkpoints(global_step)

        finally:
            self.logger.close_writers()

        print("INFO: Training Ended.")

class MCRTrainer(MUltiGPUTrainer):

    def __init__(self,
                 netD,
                 netG,
                 optD,
                 optG,
                 dataloader,
                 num_steps,
                 log_dir='./log',
                 n_dis=1,
                 lr_decay=None,
                 device=None,
                 netG_ckpt_file=None,
                 netD_ckpt_file=None,
                 print_steps=1,
                 vis_steps=500,
                 log_steps=50,
                 save_steps=5000,
                 flush_secs=30,
                 num_class=1000,
                 mode=0):

        super(MCRTrainer, self).__init__(netD, netG, optD, optG, dataloader, num_steps, log_dir, n_dis, lr_decay,
                                         device, netG_ckpt_file, netD_ckpt_file, print_steps, vis_steps, log_steps,
                                         save_steps, flush_secs)
        cfg = _C
        self.mcr_gan_loss = MCRGANloss(gam1=cfg.LOSS.GAM1, gam2=cfg.LOSS.GAM2, gam3=cfg.LOSS.GAM3, eps=cfg.LOSS.EPS, numclasses=num_class, mode=mode, rho=cfg.LOSS.RHO)

    def show(self, imgs, epoch, name):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = FF.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if not os.path.exists(f"{self.log_dir}/images"):
            os.makedirs(f"{self.log_dir}/images")
        plt.savefig(f"{self.log_dir}/images/{epoch:07d}_{name}.png", bbox_inches="tight")
        plt.close()

    def train(self):
        """
                Runs the training pipeline with all given parameters in Trainer.
                """
        # Restore models
        cfg = _C
        self.parallel = isinstance(self.netG, nn.DataParallel)

        try:
            global_step = self._restore_models_and_step()
            print("INFO: Starting training from global step {}...".format(
                global_step))

            iter_dataloader = infiniteloop(self.dataloader)
            nz = self.netD.module.nz

            start_time = time.time()
            while global_step < self.num_steps:
                log_data = metric_log.MetricLog()  # log data for tensorboard

                data_time = time.time()
                data, label = next(iter_dataloader)
                data_time = time.time() - data_time

                # Format batch and label
                real_cpu = data.to(self.device)
                if cfg.DATA.DATASET == 'cifar10_data_aug_loop':
                    real_cpu = torch.split(real_cpu, [3, 3], dim=1)
                    # print("real_cup:", real_cpu[0].size())
                    real_cpu = torch.cat(real_cpu, 0)
                    # print("real_cup:", real_cpu.size())

                real_label = label.clone().detach()

                for i in range(self.n_dis):

                    # Update Discriminator
                    self.netD.zero_grad()
                    self.optD.zero_grad()

                    # Forward pass real batch through D(X->Z)
                    Z = self.netD(real_cpu)

                    # Generate batch of latent vectors (Z->X')
                    X_bar = self.netG(torch.reshape(Z, (len(Z), nz)))

                    # Forward pass fake batch through D(X'->Z')
                    Z_bar = self.netD(X_bar.detach())

                    # Optimize Delta R(Z)+deltaR(Z')+sum(delta(R(Z,Z'))) by alternating G/D
                    errD, errD_EC = self.mcr_gan_loss(Z, Z_bar, real_label, i, self.n_dis)

                    errD.backward()
                    self.optD.step()

                # Update Discriminator
                self.netG.zero_grad()
                self.optG.zero_grad()

                # Repeat (X->Z->X'->Z')
                Z = self.netD(real_cpu)
                X_bar = self.netG(torch.reshape(Z, (len(Z), nz)))
                Z_bar = self.netD(X_bar)

                errG, errG_EC = self.mcr_gan_loss(Z, Z_bar, real_label, self.n_dis - 1, self.n_dis)

                errG = (-1) * errG
                errG.backward()
                self.optG.step()

                log_data.add_metric('errD', -errD.item(), group='discriminator loss')
                log_data.add_metric('errG', -errG.item(), group='generator loss')

                if self.mcr_gan_loss.train_mode == 0:
                    log_data.add_metric('errD_E', -errD_EC[0].item(), group='discriminator loss')
                    log_data.add_metric('errD_C', -errD_EC[1].item(), group='discriminator loss')

                    log_data.add_metric('errG_E', -errG_EC[0].item(), group='generator loss')
                    log_data.add_metric('errG_C', -errG_EC[1].item(), group='generator loss')

                elif self.mcr_gan_loss.train_mode in [1, 2]:
                    log_data.add_metric('errD_item1', -errD_EC[0].item(), group='discriminator loss')
                    log_data.add_metric('errD_item2', -errD_EC[1].item(), group='discriminator loss')
                    log_data.add_metric('errD_item3', -errD_EC[2].item(), group='discriminator loss')

                    log_data.add_metric('errG_item1', -errG_EC[0].item(), group='generator loss')
                    log_data.add_metric('errG_item2', -errG_EC[1].item(), group='generator loss')
                    log_data.add_metric('errG_item3', -errG_EC[2].item(), group='generator loss')
                elif self.mcr_gan_loss.train_mode in [10, ]:
                    nlist = [
                        'raw_deltaRz', 'raw_deltaRzbar', 'raw_sum_deltaRzzbar',
                        'aug_deltaRz', 'aug_deltaRzbar', 'aug_sum_deltaRzzbar',
                        'sum_deltaR_raw_z_aug_zbar', 'sum_deltaR_raw_z_aug_z'
                    ]
                    for i, name in enumerate(nlist):
                        log_data.add_metric('errD'+name, -errD_EC[i].item(), group='discriminator loss')
                        log_data.add_metric('errG'+name, -errG_EC[i].item(), group='generator loss')

                else:
                    raise ValueError()

                log_data = self.scheduler.step(log_data=log_data,
                                               global_step=global_step)

                # -------------------------
                #   Logging and Metrics
                # -------------------------
                if global_step % self.log_steps == 0:
                    self.logger.write_summaries(log_data=log_data,
                                                global_step=global_step)

                if global_step % self.print_steps == 0:
                    curr_time = time.time()
                    self.logger.print_log(global_step=global_step,
                                          log_data=log_data,
                                          time_taken=(curr_time - start_time) /
                                                     self.print_steps)
                    print("data load time: ", data_time)
                    print(f"[{global_step % len(self.dataloader)}/{len(self.dataloader)}]")
                    print(self.log_dir)
                    start_time = curr_time

                if global_step % self.vis_steps == 0:
                    # viz random noise
                    # self.logger.vis_images(netG=self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG,
                    #                        global_step=global_step)
                    # viz auto-encoding
                    with torch.no_grad():
                        real = self.netG(torch.reshape(Z[:64], (64, nz))).detach().cpu()
                        self.show(vutils.make_grid(real, padding=2, normalize=True), global_step, "transcript")
                        self.show(vutils.make_grid(real_cpu[:64], padding=2, normalize=True), global_step, "input")

                if global_step % self.save_steps == 0:
                    print("INFO: Saving checkpoints...")
                    self._save_model_checkpoints(global_step)

                global_step += 1

            print("INFO: Saving final checkpoints...")
            self._save_model_checkpoints(global_step)

        except KeyboardInterrupt:
            print("INFO: Saving checkpoints from keyboard interrupt...")
            self._save_model_checkpoints(global_step)

        finally:
            self.logger.close_writers()

        print("INFO: Training Ended.")

def get_dataloader(data_name, root, batch_size, num_workers):

    if data_name in ["lsun_bedroom_128", "cifar10", "stl10_48"]:
        dataset = load_dataset(root=root, name=data_name)

    elif data_name == 'celeba':
        dataset = celeba_dataset(root=root, size=128)

    elif data_name == 'mnist':

        transform = transforms.Compose(
                    [transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)])
        dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)

    elif data_name == 'TMNIST':
        transform = transforms.Compose(
                    [transforms.Resize(32),
                    transforms.ToTensor(),
                    MyAffineTransform(choices=[[0, 1], [0, 1.5], [0, 0.5], [-45, 1], [45, 1]]),
                    transforms.Normalize(0.5, 0.5)])
        dataset = datasets.MNIST(root=root, train=True,
                                    download=True, transform=transform)

    elif data_name == 'imagenet_128':
        dataset = datasets.ImageFolder(root,
                                       transform=transforms.Compose([
                                         transforms.CenterCrop(224),
                                         transforms.Resize(size=(128, 128)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                         transforms.Lambda(lambda x: x + torch.rand_like(x) / 128)
                                       ]))

    else:
        raise ValueError()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader, dataset



_C = CN()
_C.LOG_DIR = 'logs/mnist_LDR_multi'
# _C.GPUS = (0,)

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True
_C.CUDNN.WORKERS = 4

# dataset
_C.DATA = CN()
_C.DATA.ROOT = './data/'
_C.DATA.DATASET = 'mnist'
_C.DATA.IMAGE_SIZE = [32, 32]
_C.DATA.NC = 3

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.CIFAR_BACKBONE = ''
_C.MODEL.INIT = ''
_C.MODEL.L_RELU_P = 0.2
_C.MODEL.IMAGENET_WIDTH = 1024
_C.MODEL.NZ = 100  # Size of z latent vector (i.e. size of generator input)
_C.MODEL.NGF = 64  # Size of feature maps in generator
_C.MODEL.NDF = 64  # Size of feature maps in discriminator

# loss
_C.LOSS = CN()
_C.LOSS.MODE = 0  # 0 for LDR-binary, 1 for LDR multi
_C.LOSS.GAM1 = 1.
_C.LOSS.GAM2 = 1.
_C.LOSS.GAM3 = 1.
_C.LOSS.EPS = 0.5
_C.LOSS.RHO = (1.0, 1.0)

# training
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 2048
_C.TRAIN.LR_D = 0.00015
_C.TRAIN.LR_G = 0.00015
_C.TRAIN.BETA1 = 0.5  # Beta1 hyperparam for Adam optimizers
_C.TRAIN.BETA2 = 0.999  # Beta2 hyperparam for Adam optimizers
_C.TRAIN.ITERATION = 4500  # number of total iterations
_C.TRAIN.INNER_LOOP = 1
_C.TRAIN.LR_DECAY = 'linear'
_C.TRAIN.SHOW_STEPS = 100
_C.TRAIN.SAVE_STEPS = 5000

# evaluation
_C.EVAL = CN()
_C.EVAL.DATA_SAMPLE = 50000
_C.EVAL.NETD_CKPT = ''
_C.EVAL.NETG_CKPT = ''


def update_config(cfg, args):
    cfg.defrost()
    if args.cfg:
        cfg.merge_from_file(args.cfg)

    # if args.testModel:
    #     cfg.TEST.MODEL_FILE = args.testModel

    cfg.merge_from_list(args.opts)

    cfg.freeze()

def get_models(data_name, device):

    if data_name == "cifar10":
        netG, netD = get_cifar_model()
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))
    elif data_name in ['mnist', 'TMNIST']:
        netG = GeneratorMNIST().to(device)
        netG.apply(weights_init_mnist_model)
        netG = nn.DataParallel(netG)
        netD = DiscriminatorMNIST().to(device)
        netD.apply(weights_init_mnist_model)
        netD = nn.DataParallel(netD)
    elif data_name == 'stl10_48':
        netG = sngan.SNGANGenerator48().to(device)
        netG = nn.DataParallel(netG)

        netD = customSNGANDiscriminator48().to(device)
        netD = nn.DataParallel(netD)
    elif data_name in ["celeba", "lsun_bedroom_128", "imagenet_128"]:
        netG = sngan.SNGANGenerator128(ngf=cfg.MODEL.IMAGENET_WIDTH).to(device)
        netG = nn.DataParallel(netG)

        netD = customSNGANDiscriminator128(ndf=cfg.MODEL.IMAGENET_WIDTH).to(device)
        netD = nn.DataParallel(netD)
    else:
        raise ValueError()

    return netD, netG

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
config = _C
dataloader, dataset = get_dataloader(
    data_name=config.DATA.DATASET,
    root=config.DATA.ROOT,
    batch_size=config.TRAIN.BATCH_SIZE,
    num_workers=config.CUDNN.WORKERS
)

# Define models and optimizers
netD, netG = get_models(config.DATA.DATASET, device)

optD = optim.Adam(netD.parameters(), config.TRAIN.LR_D, betas=(config.TRAIN.BETA1, config.TRAIN.BETA2))
optG = optim.Adam(netG.parameters(), config.TRAIN.LR_G, betas=(config.TRAIN.BETA1, config.TRAIN.BETA2))

# Start training
trainer = MCRTrainer(
    netD=netD,
    netG=netG,
    optD=optD,
    optG=optG,
    n_dis=config.TRAIN.INNER_LOOP,
    num_steps=config.TRAIN.ITERATION,
    lr_decay=config.TRAIN.LR_DECAY,
    print_steps=config.TRAIN.SHOW_STEPS,
    vis_steps=config.TRAIN.SHOW_STEPS,
    log_steps=config.TRAIN.SHOW_STEPS,
    save_steps=config.TRAIN.SAVE_STEPS,
    dataloader=dataloader,
    log_dir=config.LOG_DIR,
    device=device,
    num_class=config.MODEL.NUM_CLASSES,
    mode=config.LOSS.MODE,
)
trainer.train()
torch.save("ctrl_disc.dat", netD)
torch.save("ctrl_gen.dat", netG)