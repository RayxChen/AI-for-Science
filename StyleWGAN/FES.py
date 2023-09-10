import torch.nn as nn
import torch.nn.functional as F


class FusedConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, 1)
        self.ln = nn.LayerNorm(out_features)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        x = self.activation(self.ln(x))
        x = x.transpose(1, 2).contiguous()
        return x


# Free Energy Surface

class FES3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = FusedConv(3, 16)
        self.l2 = FusedConv(16, 32)
        self.l3 = FusedConv(32, 64)
        self.l4 = FusedConv(64, 128)
        self.l5 = FusedConv(128, 256)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(256 * 1024, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.flatten(x)
        x = self.out(x)
        return x


class FESxz(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64 * 64, 4096)
        self.l2 = nn.Linear(4096, 2048)
        self.l3 = nn.Linear(2048, 1024)
        self.ln1 = nn.LayerNorm(4096)
        self.ln2 = nn.LayerNorm(2048)
        self.ln3 = nn.LayerNorm(1024)

        self.out = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.view(-1, 64 * 64)
        x = F.relu(self.ln1(self.l1(x)))
        x = F.relu(self.ln2(self.l2(x)))
        x = F.relu(self.ln3(self.l3(x)))
        score = self.out(x)
        return score

