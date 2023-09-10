import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import Na1024, projection, projection_xz, square_distance, get_xrd, get_dm
from model import VanillaVAE
from FES import FES3d


class VAE(pl.LightningModule):
    def __init__(self,
                 batch_size,
                 lr,
                 state,
                 debug=False,
                 xrd=False):
        super().__init__()
        self.VAE = VanillaVAE()
        self.batch_size = batch_size
        self.lr = lr
        self.state = state
        self.debug = debug
        self.xrd = xrd
        self.FES = FES3d()

    def VAE_loss(self, *args, kld_weight=1):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return loss, recons_loss, -kld_loss

    # noinspection PyTypeChecker
    def geo_penalty_3d(self, g_dmin, dmin, lamda=10):
        d = torch.where(g_dmin < dmin, dmin, torch.zeros_like(g_dmin))
        d = d.mean(dim=1).mean() * lamda
        # return F.relu(dmin-g_dmin).sum(dim=1).mean() * lamda
        return d

    def xrd_loss(self, fake_xrd, real_xrd):
        return F.l1_loss(fake_xrd, real_xrd)

    def fes_loss(self, fake_fes, real_fes, lamda=1):
        return F.mse_loss(fake_fes, real_fes) * lamda

    def sphere_loss(self, fake_xyz, real_xyz, mu, sigma, k=3, lamda=1):
        dist = torch.sqrt(torch.square(fake_xyz - real_xyz).sum(dim=1))
        return F.relu((dist - (mu + k * sigma)).sum(dim=1)).mean() * lamda

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.xrd:
            X, fes, T, dmin, xrd, mu, sigma = batch  # (B, 3, n_atoms)  # (B, 32, 32, 1)
        else:
            X, fes, T, dmin, mu, sigma = batch

        # train G
        if optimizer_idx == 0:
            fake_X, X, mu, log_var = self.VAE(X)

            VAE_loss, recons_loss, kld_loss = self.VAE_loss(fake_X, X, mu, log_var)

            sphere_loss = self.sphere_loss(fake_X, X, mu, sigma)

            # fake_fes = self.D(fake_X, mode='FES') * T
            # D_loss_fes = self.fes_loss(fake_fes, fes)

            if self.xrd:
                fake_xrd = get_xrd(fake_X)
                xrd_loss = self.xrd_loss(fake_xrd, xrd)
                self.logger.experiment.add_scalar('xrd_loss', xrd_loss, self.global_step)
            else:
                xrd_loss = 0

            # PBC dmin.max = 0.1125 box_size = 2
            box_dmin = get_dm(fake_X)

            geo_penalty = self.geo_penalty_3d(box_dmin, dmin)

            # fake_X = fake_X.transpose(1, 2).contiguous()
            #
            # if self.d2:
            #     proj_list = projection(fake_X)
            #     fake_Y_2d = self.D(proj_list, fes, '2d')
            #     G_loss_2d = self.G_loss_2d(fake_Y_2d)
            #     self.logger.experiment.add_scalar('G_loss_2d', G_loss_2d, self.global_step)
            #
            #     # fake_xz = proj_list[-1]
            # else:
            #     # fake_xz = projection_xz(fake_X)
            #     G_loss_2d = 0
            #
            # G_loss = G_loss_2d + xrd_loss + geo_penalty + sphere_loss

            loss = VAE_loss + xrd_loss + geo_penalty + sphere_loss

            self.logger.experiment.add_scalar('kld_loss', kld_loss, self.global_step)
            self.logger.experiment.add_scalar('recons_loss', recons_loss, self.global_step)
            self.logger.experiment.add_scalar('sphere_loss', sphere_loss, self.global_step)
            self.logger.experiment.add_scalar('geo_penalty', geo_penalty, self.global_step)

            return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.VAE.parameters(), self.D_lr, betas=(0.7, 0.999))
        schedule = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
        return [opt, schedule]

    def train_dataloader(self):
        dataset = Na1024(self.state, self.debug, self.xrd)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=22, pin_memory=True, shuffle=True)
        return loader
