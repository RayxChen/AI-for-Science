import torch
from model_ensemble import FES
import pytorch_lightning as pl
import numpy as np

n = 40
batch_size = 250

FES_PATH = f'./ckpt/epoch=339_data_10K_ensemble.ckpt'
config = {
    'batch_size': 200,
    'lr': 5e-4,
}

a = FES(config)
model_FES = a.load_from_checkpoint(FES_PATH, config=config)
model_FES.eval()
model_FES.cuda()

# XYZ = np.load('xyz_50K_10K.npz')['arr_0']
# T = np.load('T_50K_10K.npz')['arr_0']
# XZ = np.load('xz_50K_10K.npz')['arr_0']
# XRD = np.load('xrd_50K_10K.npz')['arr_0']
# XRD = XRD / 100

XYZ = np.load('../data/10K/xyz.npz')['arr_0']
T = np.load('../data/10K/T_01.npz')['arr_0']
XZ = np.load('../data/10K/data_2d.npz')['xz']
XRD = np.load('../data/10K/xrd_60_100.npz')['arr_0']

save = []
for i in range(n):
    xyz = XYZ[batch_size*i: batch_size*(i+1)]
    xyz = torch.from_numpy(xyz).cuda()
    xz = XZ[batch_size * i: batch_size * (i + 1)]
    xz = torch.from_numpy(xz).cuda()
    xrd = XRD[batch_size * i: batch_size * (i + 1)]
    xrd = torch.from_numpy(xrd).cuda()
    t = T[batch_size * i: batch_size * (i + 1)]
    t = torch.from_numpy(t).cuda()
    fe = model_FES([xyz, xz, xrd]) * t
    fe = fe.detach().cpu().numpy()
    save.append(fe)
save = np.concatenate(save)

print(save.shape)

np.savez_compressed('fe.npz', save)


# rand1 = torch.empty((1, 3, 1024)).normal_().cuda()
# rand2 = torch.empty((1, 3, 1024)).uniform_(-1, 1).cuda()
# s1, _ = model_FES(rand1)
# s2, _ = model_FES(rand2)

