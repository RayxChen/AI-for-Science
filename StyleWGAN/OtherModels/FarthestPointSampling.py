import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm
from utils import Na1024, Projection, get_resnet18, get_z, get_fps, Reshape


class PES(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = self._make_layers()

    def _make_layers(self):
        """xyz -> score, whether it is close to true PES"""

        layers = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Flatten(),
            nn.Linear(256 * 1024, 1),
        )

        return layers

    def forward(self, x):
        x = x.transpose(1, 2)
        PES_score = self.layers(x)

        return PES_score


class Condition(nn.Module):
    """potential, xrd spec etc. a 1D scalar"""
    def __init__(self):
        super().__init__()
        self.encoder = self._make_encoder()
        self.decoder = self._make_decoder()

    def _make_encoder(self):
        """encode condition to geometry space"""

        layers = nn.Sequential(
            nn.Linear(1, 128 * 3),
            Reshape(3, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
        )
        return layers

    def _make_decoder(self):
        """return a z_prob"""

        layers = nn.Sequential(
            nn.Linear(3, 16),
            nn.LeakyReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Flatten(),
            nn.Linear(256 * 1024, 1),
            nn.Sigmoid(),
        )

        return layers

    def forward(self, condition):
        xyz = self.encoder(condition)
        xyz = xyz.transpose(1, 2)
        z_prob = self.decoder(xyz)

        return z_prob


class CMLP(nn.Module):
    def __init__(self, point_scales, spec_norm=False):
        super().__init__()
        if spec_norm:
            SN = spectral_norm
            self.conv1 = SN(torch.nn.Conv1d(3, 64, 1))
            self.conv2 = SN(torch.nn.Conv1d(64, 64, 1))
            self.conv3 = SN(torch.nn.Conv1d(64, 128, 1))
            self.conv4 = SN(torch.nn.Conv1d(128, 256, 1))
            self.conv5 = SN(torch.nn.Conv1d(256, 512, 1))
            self.conv6 = SN(torch.nn.Conv1d(512, 1024, 1))
        else:
            self.conv1 = torch.nn.Conv1d(3, 64, 1)
            self.conv2 = torch.nn.Conv1d(64, 64, 1)
            self.conv3 = torch.nn.Conv1d(64, 128, 1)
            self.conv4 = torch.nn.Conv1d(128, 256, 1)
            self.conv5 = torch.nn.Conv1d(256, 512, 1)
            self.conv6 = torch.nn.Conv1d(512, 1024, 1)
        self.point_scales = point_scales
        self.maxpool = torch.nn.MaxPool1d(self.point_scales, 2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1((self.conv1(x))))
        x = F.relu(self.bn2((self.conv2(x))))
        x_128 = F.relu(self.bn3((self.conv3(x))))
        x_256 = F.relu(self.bn4((self.conv4(x_128))))
        x_512 = F.relu(self.bn5((self.conv5(x_256))))
        x_1024 = F.relu(self.bn6((self.conv6(x_512))))
        x_128 = self.maxpool(x_128)
        x_256 = self.maxpool(x_256)
        x_512 = self.maxpool(x_512)
        x_1024 = self.maxpool(x_1024)
        L = [x_1024, x_512, x_256, x_128]
        x = torch.squeeze(torch.cat(L, 1)) # (32, 1920)
        
        return x


class LatentFeature(nn.Module):
    def __init__(self, each_scales_size, point_scales_list, spec_norm=False):
        super().__init__()
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list

        self.CMLP1 = nn.ModuleList(
            [CMLP(point_scales=self.point_scales_list[0], spec_norm=spec_norm)
             for _ in range(self.each_scales_size)])
        self.CMLP2 = nn.ModuleList(
            [CMLP(point_scales=self.point_scales_list[1], spec_norm=spec_norm)
             for _ in range(self.each_scales_size)])
        self.CMLP3 = nn.ModuleList(
            [CMLP(point_scales=self.point_scales_list[2], spec_norm=spec_norm)
             for _ in range(self.each_scales_size)])
        self.CMLP4 = nn.ModuleList(
            [CMLP(point_scales=self.point_scales_list[3], spec_norm=spec_norm)
             for _ in range(self.each_scales_size)])

        self.conv1 = torch.nn.Conv1d(4, 1, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        outs = []
        for i in range(self.each_scales_size):
            outs.append(self.CMLP1[i](x[0]))
        for j in range(self.each_scales_size):
            outs.append(self.CMLP2[j](x[1]))
        for k in range(self.each_scales_size):
            outs.append(self.CMLP3[k](x[2]))
        for m in range(self.each_scales_size):
            outs.append(self.CMLP4[m](x[3]))
        
        z_feature = torch.stack(outs, 1)
        z_feature = F.relu(self.bn1((self.conv1(z_feature))))
        return z_feature


class Generator(nn.Module):
    def __init__(self, each_scales_size=1, point_scales_list=None):
        super().__init__()
        if point_scales_list is None:
            point_scales_list = [1024, 512, 256, 128]
        self.z_feature = LatentFeature(each_scales_size, point_scales_list, spec_norm=True)
        self.fc1 = nn.Linear(1920, 1024)    # 1024+512+256+128
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)

        self.fc1_1 = nn.Linear(1024, 1024 * 256)
        self.fc2_1 = nn.Linear(512, 512 * 128)
        self.fc3_1 = nn.Linear(256, 256 * 64)
        self.fc4_1 = nn.Linear(128, 128 * 32)

        self.convz_1 = nn.Conv1d(256, 128, 1)
        self.convz_2 = nn.Conv1d(128, 64, 1)
        self.convz_3 = nn.Conv1d(64, 32, 1)
        self.convz_4 = nn.Conv1d(32, 3, 1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.z_feature(x)
        x_1 = F.relu(self.fc1(x))  # 1024
        x_2 = F.relu(self.fc2(x_1))  # 512
        x_3 = F.relu(self.fc3(x_2))  # 256
        x_4 = F.relu(self.fc4(x_3))  # 128

        z_4 = self.fc4_1(x_4)
        z_4 = z_4.reshape(-1, 1024, 1, 4)  # 128x32
        z_3 = F.relu(self.fc3_1(x_3))
        z_3 = z_3.reshape(-1, 1024, 4, 4)  # 256x64
        z_2 = F.relu(self.fc2_1(x_2))
        z_2 = z_2.reshape(-1, 1024, 16, 4)  # 512x128
        z_1 = F.relu(self.fc1_1(x_1))
        z_1 = z_1.reshape(-1, 1024, 64, 4)  # 1024x256
        z = z_4 + z_3
        z = z.repeat([1, 1, 4, 1])
        z = z + z_2
        z = z.repeat([1, 1, 4, 1])
        z = z + z_1
        z = z.reshape(-1, 256, 1024)
        z = F.relu(self.bn1(self.convz_1(z)))
        z = F.relu(self.bn2(self.convz_2(z)))
        z = F.relu(self.bn3(self.convz_3(z)))
        xyz = torch.tanh(self.convz_4(z))

        return xyz


class Critics(nn.Module):
    def __init__(self, each_scales_size=1, point_scales_list=None):
        super().__init__()
        if point_scales_list is None:
            point_scales_list = [1024, 512, 256, 128]
        self.z_feature = LatentFeature(each_scales_size, point_scales_list, spec_norm=True)
        self.out_3d = nn.Linear(1920, 1)
        self.resnet = get_resnet18()
        self.projection = Projection.apply
        self.out_2d = nn.Linear(1024, 1)

        self.PES = PES()

    def forward(self, x, lamda=1):
        x_3d = self.z_feature(x)  # (1024 512 256 128) first 4
        x_3d_score = self.out_3d(x_3d)
        if lamda == 0:
            return x_3d_score, torch.Tensor([0])
        else:
            if len(x) > 4:
                xy, yz, xz = x[4:]
            else:
                xy, yz, xz = self.projection(x[0])  # only all atoms
            xy = self.resnet(xy)
            yz = self.resnet(yz)
            xz = self.resnet(xz)
            x_2d_score = self.out_2d(xy) + self.out_2d(yz) + self.out_2d(xz)
            return x_3d_score, x_2d_score


class GeneratorLoss(nn.Module):
    def __init__(self, lamda=1, mu=1):
        super().__init__()
        self.lamda = lamda
        self.mu = mu

    def forward(self, geometry_loss, c_p_loss):

        return geometry_loss + self.mu*c_p_loss

    def geometry_loss(self, fake_Y, fake_Y_2d):
        loss_3d = -torch.mean(fake_Y)
        loss_2d = - self.lamda * torch.mean(fake_Y_2d)
        return loss_3d, loss_2d

    @staticmethod
    def c_p_loss(c_p, fake_p):
        c_p_loss = F.l1_loss(fake_p, c_p, reduction='sum')
        return c_p_loss


class CriticsLoss(nn.Module):
    def __init__(self, lamda=1, mu=1):
        super().__init__()
        self.lamda = lamda  # for 2d
        self.mu = mu  # for c_p

    def forward(self, geometry_loss, c_p_loss):

        return geometry_loss + self.mu*c_p_loss

    def geometry_loss(self, real_Y, fake_Y, real_Y_2d, fake_Y_2d):
        loss_3d = torch.mean(fake_Y) - torch.mean(real_Y)
        loss_2d = self.lamda * (torch.mean(fake_Y_2d) - torch.mean(real_Y_2d))
        return loss_3d, loss_2d

    @staticmethod
    def c_p_loss(c_p, fake_p, real_p):
        loss_real_p = F.l1_loss(real_p, c_p, reduction='sum')
        loss_fake_p = -F.l1_loss(fake_p, c_p, reduction='sum')  # swap liquid and solid samples -> shuffle_X
        p_loss = loss_fake_p + loss_real_p
        return p_loss


class GAN(pl.LightningModule):
    def __init__(self, batch_size, D_lr, G_lr, lamda, mu):
        super().__init__()
        self.G = Generator()
        self.D = Critics()
        self.condition_potential = Condition()
        self.batch_size = batch_size
        self.G_lr = G_lr
        self.D_lr = D_lr
        self.lamda = lamda
        self.mu = mu

    def forward(self, x):
        return self.G(x)

    def G_loss(self, fake_Y, fake_Y_2d, fake_p, c_p, lamda=1, mu=1):
        generator_loss = GeneratorLoss(lamda=lamda, mu=mu)
        loss_3d, loss_2d = generator_loss.geometry_loss(fake_Y, fake_Y_2d)
        c_p_loss = generator_loss.c_p_loss(c_p, fake_p)
        geometry_loss = loss_3d + loss_2d
        G_loss = generator_loss(geometry_loss, c_p_loss)

        return G_loss, loss_3d, loss_2d, c_p_loss

    def D_loss(self, real_Y, fake_Y, real_Y_2d, fake_Y_2d, c_p, fake_p, real_p, lamda=1, mu=1):
        critic_loss = CriticsLoss(lamda=lamda, mu=mu)
        loss_3d, loss_2d = critic_loss.geometry_loss(real_Y, fake_Y, real_Y_2d, fake_Y_2d)
        c_p_loss = critic_loss.c_p_loss(c_p, fake_p, real_p)
        geometry_loss = loss_3d + loss_2d
        D_loss = critic_loss(geometry_loss, c_p_loss)

        return D_loss, loss_3d, loss_2d, c_p_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        X, xy, yz, xz, x_512, x_256, x_128, c_p, shuffle_X = batch
        xy = xy.transpose(1, 3)
        yz = yz.transpose(1, 3)
        xz = xz.transpose(1, 3)

        self.mu = 0 if self.current_epoch <= 20 else 1
        # D_lamda = 0 if self.current_epoch >= 130 else 1
        D_lamda = 0

        #train G
        if optimizer_idx == 0:
            z_prob = self.condition_potential(c_p)
            z = get_z(X, z_prob)  # (32, 3, 1024)
            z_512 = get_fps(z, 512)
            z_256 = get_fps(z_512, 256)
            z_128 = get_fps(z_256, 128)

            fake_X = self.G([z, z_512, z_256, z_128])

            fake_x_512 = get_fps(fake_X, 512)
            fake_x_256 = get_fps(fake_x_512, 256)
            fake_x_128 = get_fps(fake_x_256, 128)

            fake_Y, fake_Y_2d = self.D([fake_X, fake_x_512, fake_x_256, fake_x_128], D_lamda)

            fake_p = self.D.PES(fake_X)

            G_loss, G_3d_loss, G_2d_loss, G_p_loss = self.G_loss(fake_Y, fake_Y_2d, fake_p, c_p, self.lamda, self.mu)

            # self.log('G_loss', G_loss, on_step=True, on_epoch=True) #self.current_epoch
            self.logger.experiment.add_scalar('G_loss', G_loss, self.global_step)
            self.logger.experiment.add_scalar('G_2d_loss', G_2d_loss, self.global_step)
            self.logger.experiment.add_scalar('G_3d_loss', G_3d_loss, self.global_step)
            self.logger.experiment.add_scalar('G_p_loss', G_p_loss, self.global_step)

            return G_loss

        #train D
        if optimizer_idx == 1:
            z_prob = self.condition_potential(c_p)
            z = get_z(X, z_prob)  # (32, 3, 1024)
            z_512 = get_fps(z, 512)
            z_256 = get_fps(z_512, 256)
            z_128 = get_fps(z_256, 128)

            fake_X = self.G([z, z_512, z_256, z_128]).detach()

            fake_x_512 = get_fps(fake_X, 512)
            fake_x_256 = get_fps(fake_x_512, 256)
            fake_x_128 = get_fps(fake_x_256, 128)

            real_p = self.D.PES(X)

            fake_Y, fake_Y_2d = self.D([fake_X, fake_x_512, fake_x_256, fake_x_128], D_lamda)
            real_Y, real_Y_2d = self.D([X, x_512, x_256, x_128, xy, yz, xz])  # optimise > 4
            # always update 2d for real ones

            fake_p = c_p if self.current_epoch <= 150 else self.D.PES(shuffle_X)

            D_loss, D_3d_loss, D_2d_loss, D_p_loss= self.D_loss(real_Y, fake_Y, real_Y_2d, fake_Y_2d,
                                                            c_p, fake_p, real_p, self.lamda, self.mu)

            self.logger.experiment.add_scalar('D_loss', D_loss, self.global_step)
            self.logger.experiment.add_scalar('D_2d_loss', D_2d_loss, self.global_step)
            self.logger.experiment.add_scalar('D_3d_loss', D_3d_loss, self.global_step)
            self.logger.experiment.add_scalar('D_p_loss', D_p_loss, self.global_step)

            return D_loss

    def configure_optimizers(self):
        G_opt = torch.optim.SGD(self.G.parameters(), self.G_lr)
        D_opt = torch.optim.Adam(self.D.parameters(), self.D_lr)
        G_schedule = torch.optim.lr_scheduler.ExponentialLR(G_opt, gamma=0.99)
        D_schedule = torch.optim.lr_scheduler.ExponentialLR(D_opt, gamma=0.99)
        return [G_opt, D_opt], [G_schedule, D_schedule]

    def train_dataloader(self):
        dataset = Na1024()
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=0, pin_memory=True, shuffle=True)
        return loader



