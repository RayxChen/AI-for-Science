import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from FES import FES3d


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, type='1d', style_dim=1):
        super().__init__()
        if type == '1d':
            self.norm = nn.InstanceNorm1d(in_channel)
        if type == '2d':
            self.norm = nn.InstanceNorm2d(in_channel)

        self.type = type
        self.style = nn.Linear(style_dim, in_channel * 2)

        # self.style.bias[:in_channel] = 1
        # self.style.bias[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2)
        if self.type == '2d':
            style = style.unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, n_points = x.shape
            noise = x.new_empty(batch, 1, n_points).normal_()
        return x + self.weight * noise


class FusedAdaINBlock(nn.Module):
    def __init__(self, in_dim, out_dim, spec_norm=True, style_dim=1):
        super().__init__()
        if spec_norm:
            self.linear = spectral_norm(nn.Linear(in_dim, out_dim))
        else:
            self.linear = nn.Linear(in_dim, out_dim)
        self.AdaIN = AdaptiveInstanceNorm(out_dim, '1d', style_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.noise = NoiseInjection()

    def forward(self, x, style):
        x = self.noise(x)

        x = x.transpose(1, 2).contiguous()
        x = self.linear(x)
        x = x.transpose(1, 2).contiguous()
        x = self.AdaIN(x, style)
        x = self.activation(x)
        return x


class FusedAdaINConv(nn.Module):
    def __init__(self, in_dim, out_dim, spec_norm=True, style_dim=1, noise=False):
        super().__init__()
        if spec_norm:
            self.conv = spectral_norm(nn.Conv1d(in_dim, out_dim, 1))
        else:
            self.conv = nn.Conv1d(in_dim, out_dim, 1)
        self.AdaIN = AdaptiveInstanceNorm(out_dim, '1d', style_dim)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.noise = NoiseInjection() if noise else None

    def forward(self, x, style):
        if self.noise is not None:
            x = self.noise(x)
        x = self.conv(x)
        x = self.AdaIN(x, style)
        x = self.activation(x)
        return x


class AdaMLP(nn.Module):
    def __init__(self, spec_norm=True, style_dim=1):
        super().__init__()

        self.l1 = FusedAdaINConv(3, 32, spec_norm, style_dim)
        self.l2 = FusedAdaINConv(32, 64, spec_norm, style_dim)
        self.l3 = FusedAdaINConv(64, 128, spec_norm, style_dim)
        self.l4 = FusedAdaINConv(128, 256, spec_norm, style_dim)
        self.l5 = FusedAdaINConv(256, 512, spec_norm, style_dim)
        self.l6 = FusedAdaINConv(512, 1024, spec_norm, style_dim)

        self.maxpool = torch.max

    def forward(self, x, w):

        x_32 = self.l1(x, w)
        x_64 = self.l2(x_32, w)
        x_128 = self.l3(x_64, w)
        x_256 = self.l4(x_128, w)
        x_512 = self.l5(x_256, w)
        x_1024 = self.l6(x_512, w)

        x = [x_1024, x_512, x_256, x_128, x_64, x_32]  #
        x = torch.cat(x, dim=1)
        x = self.maxpool(x, 2)[0]
        return x


class Generator(nn.Module):
    def __init__(self, spec_norm=True, style_dim=1, noise=True):
        super().__init__()
        self.z_feature = AdaMLP(spec_norm, style_dim)
        self.fc1 = nn.Linear(1920+96, 1024 * 256)  # 1024+512+256+128
        # self.fc1_1 = nn.Linear(1024, 1024 * 256)

        # self.convz_0 = FusedAdaINConv(512, 256, spec_norm, style_dim, noise)
        self.convz_1 = FusedAdaINConv(256, 128, spec_norm, style_dim, noise)
        self.convz_2 = FusedAdaINConv(128, 64, spec_norm, style_dim, noise)
        self.convz_3 = FusedAdaINConv(64, 32, spec_norm, style_dim, noise)
        self.convz_4 = nn.Conv1d(32, 3, 1)

    def forward(self, x, w):

        x = self.z_feature(x, w)

        x = F.relu(self.fc1(x))  # 1024
        # x = F.relu(self.fc1_1(x))  # 1024

        x = x.reshape(-1, 256, 1024)

        x = self.convz_1(x, w)
        x = self.convz_2(x, w)
        x = self.convz_3(x, w)
        xyz = torch.tanh(self.convz_4(x))

        return xyz


class FusedConv(nn.Module):
    def __init__(self, in_channel, out_channel, spec_norm=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, 1)
        if spec_norm:
            self.conv = spectral_norm(self.conv)
        self.bn = nn.InstanceNorm1d(out_channel)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        return x


class Critic3d(nn.Module):
    def __init__(self, spec_norm=False, style_dim=1):
        super().__init__()
        self.z_feature = AdaMLP(spec_norm, style_dim)
        self.geo_3d = nn.Linear(1920+96, 1)

    def forward(self, x, w):

        x = self.z_feature(x, w)
        geo_score = self.geo_3d(x)
        return geo_score


class Img2Score1(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Linear(64*64, 1)

    def forward(self, x):
        # x = x.squeeze()
        x = x.view(-1, 64 * 64)
        x = self.score(x)
        return x


class Critic2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.xy = Img2Score1()
        self.yz = Img2Score1()
        self.xz = Img2Score1()

    def forward(self, xyz):
        xy, yz, xz = xyz
        xy = self.xy(xy)
        yz = self.yz(yz)
        xz = self.xz(xz)
        return xy + yz + xz


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic3d = Critic3d()
        self.critic2d = Critic2d()
        self.FES = FES3d()

    def forward(self, fake_X, w=None, mode='3d'):
        if mode == '2d':
            return self.critic2d(fake_X)
        if mode == '3d':
            return self.critic3d(fake_X, w)
        if mode == 'FES':
            return self.FES(fake_X)

