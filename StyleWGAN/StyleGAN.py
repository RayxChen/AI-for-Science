import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import Generator, Critic
from utils import Na1024, projection, projection_xz, square_distance, get_xrd, get_dm, get_pbc
# from pytorch3d.loss import chamfer_distance


class GAN(pl.LightningModule):
    def __init__(self,
                 batch_size,
                 D_lr, G_lr,
                 state,
                 debug=False,
                 xrd=False, d2=False):
        super().__init__()
        self.G = Generator()
        self.D = Critic()
        self.batch_size = batch_size
        self.G_lr = G_lr
        self.D_lr = D_lr
        self.state = state
        self.debug = debug
        self.xrd = xrd
        self.d2 = d2

    def forward(self, x):
        return self.G(x)

    def G_loss(self, fake_Y):
        loss_3d = -torch.mean(fake_Y)
        return loss_3d

    def G_loss_2d(self, fake_Y_2d, lamda=1):
        loss_2d = -torch.mean(fake_Y_2d)
        return loss_2d * lamda

    def D_loss_2d(self, real_Y_2d, fake_Y_2d, lamda=1):
        loss_2d = torch.mean(fake_Y_2d) - torch.mean(real_Y_2d)
        return loss_2d * lamda

    # def D_loss_2d(self, real_Y_2d, fake_Y_2d, margin=0.2, lamda=1):
    #     loss_2d = F.relu((fake_Y_2d - real_Y_2d).abs().mean() - margin)
    #     return loss_2d * lamda

    def D_loss(self, real_Y, fake_Y):
        loss_3d = torch.mean(fake_Y) - torch.mean(real_Y)
        return loss_3d

    def chamfer_loss(self, fake, real):
        B, C, N = real.shape
        dist = square_distance(fake, real)
        S1 = torch.sum(torch.min(dist, dim=1)[0], 1)
        S2 = torch.sum(torch.min(dist, dim=2)[0], 1)
        S = torch.mean((S1 + S2) / N)
        return S

    def geo_penalty_3d(self, g_dmin, dmin, lamda=50):
        d = torch.where(g_dmin < dmin, dmin, torch.zeros_like(g_dmin))
        d = d.mean(dim=1).mean() * lamda
        # return F.relu(dmin-g_dmin).sum(dim=1).mean() * lamda
        return d

    def xrd_loss(self, fake_xrd, real_xrd):
        return F.l1_loss(fake_xrd, real_xrd)

    def fes_loss(self, fake_fes, real_fes, lamda=1):
        return F.mse_loss(fake_fes, real_fes) * lamda

    def sphere_loss(self, fake_xyz, real_xyz, mu, sigma, k=1, lamda=5):
        dist = torch.sqrt(torch.square(fake_xyz - real_xyz).sum(dim=1))
        return F.relu((dist - (mu + k * sigma)).sum(dim=1)).mean() * lamda

    def get_z(self, X, p_prob, k=0.01):
        p_prob = p_prob.unsqueeze(2) * torch.ones((X.size(0), X.size(1), 1)).to(X.device)
        delta = torch.normal(0, k * p_prob).to(X.device)
        mask = torch.logical_and(X + delta >= -1, X + delta <= 1)
        z = torch.where(mask, X + delta, X)
        return z

    # def get_z(self, fes):
    #     z_id = torch.empty(fes.size(0), 2 - 1, requires_grad=False).normal_().to(fes.device)
    #     z = torch.cat([z_id, fes], dim=1)
    #     return z

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.xrd:
            X, xy, yz, xz, fes, T, dmin, xrd, mu, sigma = batch  # (B, 3, n_atoms)  # (B, 32, 32, 1)
        else:
            X, xy, yz, xz, fes, T, dmin, mu, sigma = batch

        # train G
        if optimizer_idx == 0:
            z = self.get_z(X, fes)

            fake_X = self.G(z, fes)  # (B, 3, 1024)
            fake_Y_3d = self.D(fake_X, fes)
            G_loss_3d = self.G_loss(fake_Y_3d)

            sphere_loss = self.sphere_loss(fake_X, X, mu, sigma)

            fake_fes = self.D(fake_X, mode='FES') * T
            D_loss_fes = self.fes_loss(fake_fes, fes)

            if self.xrd:
                fake_xrd = get_xrd(fake_X)
                xrd_loss = self.xrd_loss(fake_xrd, xrd)
                self.logger.experiment.add_scalar('xrd_loss', xrd_loss, self.global_step)
            else:
                xrd_loss = 0

            # PBC dmin.max = 0.1125 box_size = 2
            box_dmin = get_dm(fake_X)
            # pbc_x, pbc_y, pbc_z = get_pbc(fake_X)
            # pbc_xmin = get_dm(pbc_x)
            # pbc_ymin = get_dm(pbc_y)
            # pbc_zmin = get_dm(pbc_z)
            geo_penalty = self.geo_penalty_3d(box_dmin, dmin)
                          # + self.geo_penalty_3d(pbc_xmin, dmin) \
                          # + self.geo_penalty_3d(pbc_ymin, dmin) \
                          # + self.geo_penalty_3d(pbc_zmin, dmin)

            fake_X = fake_X.transpose(1, 2).contiguous()

            if self.d2:
                proj_list = projection(fake_X)
                fake_Y_2d = self.D(proj_list, fes, '2d')
                G_loss_2d = self.G_loss_2d(fake_Y_2d)
                self.logger.experiment.add_scalar('G_loss_2d', G_loss_2d, self.global_step)

                # fake_xz = proj_list[-1]
            else:
                # fake_xz = projection_xz(fake_X)
                G_loss_2d = 0

            # fake_fes = self.D(fake_X, mode='FES') * T
            # D_loss_fes = self.fes_loss(fake_fes, fes)

            # chamfer_loss, _ = chamfer_distance(X.transpose(1, 2), fake_X)

            G_loss = G_loss_2d + G_loss_3d \
                     + xrd_loss + geo_penalty + sphere_loss + D_loss_fes # + chamfer_loss

            self.logger.experiment.add_scalar('G_loss_3d', G_loss_3d, self.global_step)
            self.logger.experiment.add_scalar('sphere_loss', sphere_loss, self.global_step)
            # self.logger.experiment.add_scalar('chamfer_loss', chamfer_loss, self.global_step)
            self.logger.experiment.add_scalar('D_loss_fes', D_loss_fes, self.global_step)
            self.logger.experiment.add_scalar('geo_penalty', geo_penalty, self.global_step)

            return G_loss

        # train D
        if optimizer_idx == 1:
            z = self.get_z(X, fes)

            with torch.no_grad():
                fake_X = self.G(z, fes)

            fake_Y_3d = self.D(fake_X, fes)
            real_Y_3d = self.D(X, fes)
            D_loss_3d = self.D_loss(real_Y_3d, fake_Y_3d)

            if self.d2:
                fake_X = fake_X.transpose(1, 2).contiguous()
                proj_list = projection(fake_X)
    
                fake_Y_2d = self.D(proj_list, fes, '2d')
                real_Y_2d = self.D([xy, yz, xz], fes, '2d')
                D_loss_2d = self.D_loss_2d(real_Y_2d, fake_Y_2d)
            else:
                D_loss_2d = 0


            D_loss = D_loss_2d + D_loss_3d

            # self.logger.experiment.add_scalar('D_loss', D_loss, self.global_step)
            self.logger.experiment.add_scalar('D_loss_3d', D_loss_3d, self.global_step)
            self.logger.experiment.add_scalar('D_loss_2d', D_loss_2d, self.global_step)

            return D_loss

    def configure_optimizers(self):
        G_opt = torch.optim.SGD(self.G.parameters(), self.G_lr)
        D_opt = torch.optim.Adam(self.D.parameters(), self.D_lr, betas=(0.7, 0.999))
        G_schedule = torch.optim.lr_scheduler.ExponentialLR(G_opt, gamma=0.99)
        D_schedule = torch.optim.lr_scheduler.ExponentialLR(D_opt, gamma=0.99)
        return [G_opt, D_opt], [G_schedule, D_schedule]

    def train_dataloader(self):
        dataset = Na1024(self.state, self.debug, self.xrd)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=22, pin_memory=True, shuffle=True)
        return loader
