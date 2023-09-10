import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def knn(x, k: int):
    """
    inputs:
    - x: b x npoints1 x num_dims (partical_cloud)
    - k: int (the number of neighbor)
    outputs:
    - idx: int (neighbor_idx)
    """
    # x : (batch_size, feature_dim, num_points)
    # Retrieve nearest neighbor indices

    if torch.cuda.is_available():
        from knn_cuda import KNN

        ref = x.transpose(2, 1).contiguous()  # (batch_size, num_points, feature_dim)
        query = ref
        _, idx = KNN(k=k, transpose_mode=True)(ref, query)

    else:
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx


def get_graph_feature(x, k: int = 6, idx=None):
    """
    inputs:
    - x: b x npoints1 x num_dims (partical_cloud)
    - k: int (the number of neighbor)
    - idx: neighbor_idx
    outputs:
    - feature: b x npoints1 x (num_dims*2)
    """

    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = idx.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class SELayer(nn.Module):
    """
    input:
        x:(b, c, m, n)
    output:
        out:(b, c, m', n')
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, type='1d', style_dim=1):
        super().__init__()
        if type == '1d':
            self.norm = nn.InstanceNorm1d(in_channel)
        if type == '2d':
            self.norm = nn.InstanceNorm2d(in_channel)

        self.type = type
        self.style = nn.Linear(style_dim, in_channel * 2)

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2)
        if self.type == '2d':
            style = style.unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class FusedAdaINSEConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, SE_layer=True, spec_norm=True, style_dim=1):
        super().__init__()
        if spec_norm:
            self.conv = spectral_norm(nn.Conv2d(in_dim, out_dim, 1, bias=False))
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, 1, bias=False)

        self.SE = SELayer(out_dim) if SE_layer else None
        self.AdaIN = AdaptiveInstanceNorm(out_dim, '2d', style_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        x = self.conv(x)
        x = self.AdaIN(x, style)
        if self.SE:
            x = self.SE(x)
        x = self.activation(x)
        return x
    
    
class FusedAdaINConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, spec_norm=True, style_dim=1):
        super().__init__()
        if spec_norm:
            self.conv = spectral_norm(nn.Conv1d(in_dim, out_dim, 1, bias=False))
        else:
            self.conv = nn.Conv1d(in_dim, out_dim, 1, bias=False)
        self.AdaIN = AdaptiveInstanceNorm(out_dim, '1d', style_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, style):
        x = self.conv(x)
        x = self.AdaIN(x, style)
        x = self.activation(x)
        return x

# a = torch.empty(2, 80, 10, 10).uniform_(0, 1)
# s = torch.empty(2, 1).uniform_(0, 1)
# b = FusedAdaINSEConv2d(80, 40)(a, s)
# print(b.shape)


class EdgeConv(nn.Module):
    """
    input
    - point_cloud: b x num_dims x npoints1
    output
    - feture:  b x feature_size
    """

    def __init__(
        self,
        spec_norm: bool = True,
        use_SElayer: bool = True,
        k: int = 6,
        latent_dim: int = 1024,
        style_dim: int = 1,
    ):
        super().__init__()
        self.k = k
        self.latent_dim = latent_dim

        # 1024/16 = 64
        self.ada_conv1 = FusedAdaINSEConv2d(6, self.latent_dim // 16,
                                           use_SElayer, spec_norm, style_dim)
        self.ada_conv2 = FusedAdaINSEConv2d(self.latent_dim // 8, self.latent_dim // 8,
                                           use_SElayer, spec_norm, style_dim)
        self.ada_conv3 = FusedAdaINSEConv2d(self.latent_dim // 4, self.latent_dim // 4,
                                           use_SElayer, spec_norm, style_dim)
        self.ada_conv4 = FusedAdaINSEConv2d(self.latent_dim // 2, self.latent_dim // 2,
                                           use_SElayer, spec_norm, style_dim)

        # self.out_conv = FusedAdaINConv1d(self.latent_dim * 15 // 16, self.latent_dim,
        #                                  spec_norm, style_dim)

    def forward(self, x, style):
        # (B, 3, n_points)

        x = get_graph_feature(x, k=self.k)
        x = self.ada_conv1(x, style)
        x1 = x.max(dim=-1)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.ada_conv2(x, style)
        x2 = x.max(dim=-1)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.ada_conv3(x, style)
        x3 = x.max(dim=-1)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.ada_conv4(x, style)
        x4 = x.max(dim=-1)[0]

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = x.max(dim=-1)[0]  # (B, 960)

        return x


# a = torch.empty(2, 3, 1024).uniform_(0, 1).cuda()
# s = torch.empty(2, 1).uniform_(0, 1).cuda()
# b = EdgeConv(k=6, use_SElayer=True).cuda()(a, s)
# # #
# print(b.shape)