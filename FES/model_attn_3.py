import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import get_data
import matplotlib.pyplot as plt


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        # self.trans_conv = nn.Conv1d(channels, channels, 1)
        # self.after_norm = nn.InstanceNorm1d(channels)
        # self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dk = channels**0.5

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)    # b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        attention = attention / self.dk
        x = torch.bmm(x_v, attention)
        # x_r = torch.bmm(x_v, attention)  # b, c, n
        # x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        # x = x + x_r
        return x


class LatentFeature(nn.Module):
    def __init__(self, inputd_dim=3, hidden_size=128):
        super().__init__()
        # self.hidden_size = hidden_size
        self.l1 = nn.Conv1d(inputd_dim, hidden_size, 1, bias=False)
        self.l2 = nn.Conv1d(hidden_size, hidden_size, 1, bias=False)
        self.bn1 = nn.InstanceNorm1d(hidden_size)
        self.bn2 = nn.InstanceNorm1d(hidden_size)
        self.dist_key = nn.Sequential(
            nn.Conv1d(1024, 256, 1, bias=False),
            nn.InstanceNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, hidden_size, 1, bias=False),
        )
        self.sa1 = SA_Layer(hidden_size)
        self.sa2 = SA_Layer(hidden_size)
        self.sa3 = SA_Layer(hidden_size)
        self.sa4 = SA_Layer(hidden_size)
        # self.out = nn.Linear(hidden_size * 4, 1)

    def forward(self, x):
        x_k = x.transpose(1, 2)
        x_k = torch.cdist(x_k, x_k)
        x_k = self.dist_key(x_k)
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)

        x3 = self.sa3(x_k)
        x4 = self.sa4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.max(x, 2)[0]
        # x = x.view(-1, self.hidden_size * 4)
        # x = self.out(x)
        return x


# class Attention(nn.Module):
#     def __init__(self, in_channels, n_heads=4):
#         super().__init__()
#         self.Q = nn.Linear(in_channels, in_channels//n_heads)
#         self.K = nn.Linear(in_channels, in_channels//n_heads)
#         self.V = nn.Linear(in_channels, in_channels//n_heads)
#         self.dk = in_channels ** 0.5
#
#     def forward(self, x):
#
#         x_q = self.Q(x)  # (B, C, N) -> (B, N, C)
#         x_k = self.K(x).transpose(1, 2)  # (B, C, N)
#         x_v = self.V(x)
#         energy = torch.bmm(x_q, x_k)  # (B, N, N)
#         attention = F.softmax(energy, dim=-1)
#         # attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
#         attention = attention / self.dk
#         x = torch.bmm(attention, x_v)  # (B, C, N)
#
#         return x, attention


# class MultiHeadAttention(nn.Module):
#     def __init__(self, in_channels, n_heads=4, dropout=0.1):
#         super().__init__()
#         assert in_channels % n_heads == 0
#         self.attentions = [Attention(in_channels, n_heads) for _ in range(n_heads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#         self.linear = nn.Linear(in_channels, in_channels)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(in_channels)
#         self.combine_weight = nn.Linear(n_heads, 1)
#
#     def forward(self, x, weight=True):
#         attn = [att(x) for att in self.attentions]
#
#         context = torch.cat([att[0] for att in attn], dim=-1)
#
#         out = self.linear(context)
#         out = self.dropout(out)
#         out = self.layer_norm(x + out)
#
#         if weight:
#             attn_weight = torch.stack([att[1] for att in attn], dim=-1)
#             # import IPython; IPython.embed()
#             attn_weight = self.combine_weight(attn_weight).squeeze()
#             return out, attn_weight
#         else:
#             return out


class FES(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.Feature3D = LatentFeature(3, 64)
        self.FeatureXY = LatentFeature(2, 32)
        self.FeatureYZ = LatentFeature(2, 32)
        self.FeatureXZ = LatentFeature(2, 32)
        self.Fuse2d = nn.Linear(3*128, 256)
        self.bn = nn.BatchNorm1d(256)
        self.out = nn.Sequential(nn.Linear(4*64 + 256 + 60, 512),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1)
                                 )

        # self.attention = config['attention']
        self.lr = config['lr']
        self.train_data, self.valid_data, self.test_data = get_data(config['batch_size'])

    def forward(self, x):
        xyz, xy, yz, xz, xrd = x
        # print(xyz.shape)
        # print(xy.shape)
        xyz_3df = self.Feature3D(xyz)
        xy_f = self.FeatureXY(xy)
        yz_f = self.FeatureYZ(yz)
        xz_f = self.FeatureXZ(xz)
        xyz_2df = torch.cat([xy_f, yz_f, xz_f], dim=1)
        xyz_2df = F.relu(self.bn(self.Fuse2d(xyz_2df)))
        feature = torch.cat([xyz_3df, xyz_2df, xrd], dim=1)
        # print(feature.shape)
        x = self.out(feature)
        return x

    def plot_attn(self, attn, i):
        fig = plt.figure()
        norm = plt.Normalize(vmin=attn.min(), vmax=attn.max())
        # norm = plt.Normalize(vmin=-1, vmax=1)
        image = plt.imshow(attn, norm=norm, cmap=plt.cm.gray)
        fig.colorbar(image)
        if not os.path.exists('./attn'):
            os.mkdir('./attn')
        path = os.path.join('./attn', f'{self.current_epoch}_{i}.png')
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    def loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def training_step(self, batch, *args):
        xyz, xrd, fes, T = batch
        x, y, z = torch.unbind(xyz, dim=1)
        xy = torch.stack([x, y], dim=1)
        yz = torch.stack([y, z], dim=1)
        xz = torch.stack([x, z], dim=1)
        X = [xyz, xy, yz, xz, xrd]
        E = self.forward(X)

        fes_hat = E * T
        loss = self.loss(fes_hat, fes)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, *args):
        xyz, xrd, fes, T = batch
        x, y, z = torch.unbind(xyz, dim=1)
        xy = torch.stack([x, y], dim=1)
        yz = torch.stack([y, z], dim=1)
        xz = torch.stack([x, z], dim=1)
        X = [xyz, xy, yz, xz, xrd]
        E = self.forward(X)

        fes_hat = E * T
        loss = self.loss(fes_hat, fes)
        self.log("valid_loss", loss)

        return loss

    def test_step(self, batch, *args):
        xyz, xrd, fes, T = batch
        x, y, z = torch.unbind(xyz, dim=1)
        xy = torch.stack([x, y], dim=1)
        yz = torch.stack([y, z], dim=1)
        xz = torch.stack([x, z], dim=1)
        X = [xyz, xy, yz, xz, xrd]
        E = self.forward(X)

        fes_hat = E * T
        loss = self.loss(fes_hat, fes)
        self.log("test_loss", loss)

        return loss

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.valid_data

    def test_dataloader(self):
        return self.test_data

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.7, 0.999))
        schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [schedule]
