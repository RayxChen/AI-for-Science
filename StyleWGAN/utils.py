import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
# import torch.multiprocessing as mp
from FES import FESxz

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    # PulsarPointsRenderer,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor
)


class Na1024(Dataset):
    def __init__(self, state, debug=False, xrd=False):
        if debug:
            data_dir = '../data_debug'
        else:
            if state == 'solid':
                data_dir = '../data_solid'
            elif state == 'liquid':
                data_dir = '../data_liquid'
            elif state == 'transition':
                data_dir = '../data_transition'
            else:
                raise ValueError('wrong input state')
        xyz = np.load(f'{data_dir}/xyz.npz', allow_pickle=True)['arr_0']
        xy, yz, xz = np.load(f'{data_dir}/data_2d.npz', allow_pickle=True).values()
        FES = np.load(f'{data_dir}/FES_01.npz')['arr_0']
        T = np.load(f'{data_dir}/T_01.npz')['arr_0']
        mu, sigma = np.load(f'{data_dir}/sphere.npz', allow_pickle=True).values()

        dmin = np.load(f'{data_dir}/dmin.npz')['arr_0']
        self.dmin = torch.from_numpy(dmin)

        if xrd:
            xrd = np.load(f'{data_dir}/xrd.npz')['arr_0']
            self.xrd = torch.from_numpy(xrd)
        else:
            self.xrd = None

        self.FES = torch.from_numpy(FES)
        self.xyz = torch.from_numpy(xyz)
        self.xy = torch.from_numpy(xy)
        self.yz = torch.from_numpy(yz)
        self.xz = torch.from_numpy(xz)
        self.T = torch.from_numpy(T)
        self.mu = torch.from_numpy(mu)
        self.sigma = torch.from_numpy(sigma)

        self.n_samples = len(xyz)

    def __getitem__(self, idx):
        if self.xrd is not None:
            return self.xyz[idx], self.xy[idx], self.yz[idx], self.xz[idx],\
                   self.FES[idx], self.T[idx], self.dmin[idx], self.xrd[idx], self.mu[idx], self.sigma[idx]
        else:
            return self.xyz[idx], self.xy[idx], self.yz[idx], self.xz[idx],\
                   self.FES[idx], self.T[idx], self.dmin[idx], self.mu[idx], self.sigma[idx]

    def __len__(self):
        return self.n_samples


def get_fps_list(x):
    x_512 = get_fps(x, 512)
    x_256 = get_fps(x_512, 256)
    return [x, x_512, x_256]


def get_pbc(xyz, box_size=2, margin=0.94):
    pbc_xyz = torch.where(torch.abs(xyz) > margin, xyz, torch.zeros_like(xyz))
    x, y, z = torch.unbind(pbc_xyz, dim=1)

    pbc_x = torch.where(-x > margin, x+box_size, x)
    pbc_y = torch.where(-y > margin, y+box_size, y)
    pbc_z = torch.where(-z > margin, z+box_size, z)

    pbc_x = torch.stack([pbc_x, y, z], dim=1)
    pbc_y = torch.stack([x, pbc_y, z], dim=1)
    pbc_z = torch.stack([x, y, pbc_z], dim=1)

    return pbc_x, pbc_y, pbc_z


def get_dm(xyz):
    xyz = xyz.transpose(1, 2).contiguous()
    R_ij = torch.cdist(xyz, xyz)
    # dmax = R_ij.max(dim=1)[0]
    # dmax = dmax.max(dim=1, keepdim=True)[0]
    dmin = torch.topk(R_ij, k=2, dim=1, largest=False)[0]
    # dmin = dmin[:, -1].min(dim=1, keepdim=True)[0]
    dmin = dmin[:, -1]

    return dmin


def get_xrd(xyz, qhist=60, theta_min=10, theta_max=40, lamda=1.5406, maxR=8.8):
    device = xyz.device
    xyz = xyz.transpose(1, 2)  # B, C, N -> B, N, C
    B, N, C = xyz.size()
    xyz = 35.2 * (xyz + 1) / 2
    theta_step = (theta_max - theta_min) / qhist
    theta = torch.arange(theta_min, theta_max, theta_step).view(1, qhist, 1).to(device)
    pi = 3.141592

    q = 4 * pi * torch.sin(theta * pi / 180) / lamda
    fij_Na = 4.7626 * torch.exp(-3.2850 * (q / (4 * pi)) * (q / (4 * pi))) + \
             3.1736 * torch.exp(-8.8422 * (q / (4 * pi)) * (q / (4 * pi))) + \
             1.2674 * torch.exp(-0.3136 * (q / (4 * pi)) * (q / (4 * pi))) + \
             1.1128 * torch.exp(-129.424 * (q / (4 * pi)) * (q / (4 * pi))) + 0.676

    R_ij = torch.cdist(xyz, xyz)

    R_ij = R_ij[:, torch.triu(torch.ones(N, N), diagonal=1) == 1].flatten(start_dim=1, end_dim=-1)
    R_ij = torch.where(R_ij < maxR, R_ij, torch.zeros_like(R_ij))
    R_ij = R_ij.reshape(B, 1, -1)

    v = torch.where(R_ij != 0,
                    torch.sin(q * R_ij) / (q * R_ij) * torch.sin(pi * R_ij / maxR) / (pi * R_ij / maxR) / N,
                    R_ij)

    v = torch.einsum('bqi->bq', v)
    fij_Na = fij_Na.squeeze()
    intensity = fij_Na * fij_Na * (2*v + 1)

    return intensity

# def get_z(X, c_p_prob, c_xrd_prob=None):
#     c_p_prob = c_p_prob.unsqueeze(2)
#     c_p_prob = c_p_prob * torch.ones((X.size(0), 1, X.size(2))).to(X.device)
#
#     if c_xrd_prob is not None:
#         prob = c_xrd_prob * c_p_prob
#     else:
#         prob = c_p_prob
#
#     mask = torch.bernoulli(prob).bool().to(X.device)
#     z = torch.empty(X.size()).uniform_(-1, 1).to(X.device)
#     z = torch.where(mask, z, X)
#     return z

def get_z(X, p_prob):
    p_prob = p_prob.unsqueeze(2) * torch.ones((X.size(0), X.size(1), 1)).to(X.device)
    delta = torch.normal(0, p_prob).to(X.device)
    mask = torch.logical_and(X+delta >= -1, X+delta <= 1)
    z = torch.where(mask, X+delta, X)
    return z


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """

    B, N, _ = src.shape
    _, M, _ = dst.shape
    # print ('before matmul size', src.size(), dst.size())
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def get_fps(xyz, npoint, zeros=True):   # farthest_point_sample
    device = xyz.device
    xyz = xyz.transpose(1, 2)
    B, N, C = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if zeros:
        farthest = torch.zeros(B, dtype=torch.long).to(device)
    else:
        farthest = torch.ones(B, dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = batch_indices.view(-1, 1).repeat(repeat_shape)
    new_points = xyz[batch_indices, idx, :]
    new_points = new_points.transpose(1, 2)
    return new_points


class Viusal(Callback):
    def __init__(self, directory, epoch=0, kind='solid'):
        if not os.path.exists(directory):
            os.mkdir(directory)
        self.directory = directory
        self.epoch = epoch
        # self.FES = load_checkpoint(FES3d(), '../PES_attn/ckpt/epoch=149-v1.ckpt')
        # for param in self.FES.parameters():
        #     param.requires_grad = False

        xyz = np.load(f'../data_{kind}/xyz.npz')['arr_0']
        k = 0.01
        l = len(xyz)
        if kind == 'solid':
            T_0 = 370 - k*l
        elif kind == 'liquid':
            T_0 = 370 + k*l
        a, b, c = int(0.2 * l), int(0.5 * l), int(0.8 * l)

        self.data = np.take(xyz, [a, b, c], axis=0)
        self.fes = np.take(np.load(f'../data_{kind}/FES_01.npz')['arr_0'], [a, b, c], axis=0)
        self.T_01 = np.take(np.load(f'../data_{kind}/T_01.npz')['arr_0'], [a, b, c], axis=0)
        self.T = np.array([T_0 + k*a, T_0 + k*b, T_0 + k*c])[:, None].astype('float32')
        del xyz

    def on_epoch_end(self, trainer, pl_module):
        device = pl_module.device
        data = torch.tensor(self.data).to(device)
        fes = torch.tensor(self.fes).to(device)
        T = torch.tensor(self.T).to(device)
        T_01 = torch.tensor(self.T_01).to(device)
        # self.FES.to(device)

        with torch.no_grad():
            z = pl_module.get_z(data, fes)
            fake_X = pl_module.G(z, fes)

            # fake_X = pl_module.G(data, fes)

            # pred_fes = self.FES(fake_X)*T_01

            pred_fes = pl_module.D(fake_X, mode='FES') * T_01

            g = fake_X.transpose(1, 2).cpu()
            T = T.cpu().tolist()
            fes = fes.cpu().tolist()
            pred_fes = pred_fes.cpu().tolist()

            # fake_X = fake_X.transpose(1, 2).contiguous()
            # fake_xz = projection_xz(fake_X)
            # pred_fes = pl_module.D(fake_xz, mode='FES') * T_01
            #
            # g = fake_X.cpu()
            # T = T.cpu().tolist()
            # fes = fes.cpu().tolist()
            # pred_fes = pred_fes.cpu().tolist()

            if self.epoch % 5 == 0:
                # processes = []
                # for task in (self.plot_dist, self.plot_2d, self.plot_25d):
                #     t = mp.Process(target=task, args=(g, z_p, pred_fes))
                #     t.start()
                #     processes.append(t)
                # [p.join() for p in processes]

                for i in range(len(self.T)):
                    self.plot_dist(g[i], T[i][0], fes[i][0], pred_fes[i][0], i)
                    self.plot_2d(g[i], T[i][0], fes[i][0], pred_fes[i][0], i)
                    self.plot_25d(g[i], T[i][0], fes[i][0], pred_fes[i][0], i)

            self.epoch += 1

    def plot_dist(self, g, T, fes, pred_fes, i):
        fig1, ax = plt.subplots(1, 1, num=0, figsize=(6, 4))
        sns.set(style='darkgrid', palette='muted', color_codes=True)
        sns.distplot(g, ax=ax, bins=20, hist_kws={'edgecolor': 'k'})
        ax.set_title(f'generated_{self.epoch}_{i}')
        ax.set_xlim([-1, 1])

        fig1.text(0.15, -0.05, f'T={T:.2f} | fes={fes:.2f} | pred_fes={pred_fes:.2f}', size=15)

        path = os.path.join(self.directory, f'dist_{self.epoch}_{i}.png')
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    def plot_2d(self, g, T, fes, pred_fes, i):
        fig2, axes = plt.subplots(1, 3, num=1, figsize=(12, 4))
        axes.flatten()
        axes[0].scatter(g[:, 0], g[:, 1], s=12, alpha=0.3, c='r')
        axes[1].scatter(g[:, 1], g[:, 2], s=12, alpha=0.3, c='g')
        axes[2].scatter(g[:, 2], g[:, 0], s=12, alpha=0.3, c='b')

        fig2.text(0.25, -0.05, f'T={T:.2f} | fes={fes:.2f} | pred_fes={pred_fes:.2f}', size=15)

        path = os.path.join(self.directory, f'xyz_{self.epoch}_{i}.png')
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    def plot_25d(self, g, T, fes, pred_fes, i):
        fig25 = plt.figure(figsize=(10, 5), num=2)
        ax1 = fig25.add_subplot(121, projection='3d', elev=0, azim=45)
        ax1.scatter3D(g[:, 0], g[:, 1], g[:, 2], s=5, c='royalblue', alpha=0.5)
        ax1.set_xticks([])
        ax1.set_yticks([])
        fig25.text(0.2, -0.05, f'T={T:.2f} |fes={fes:.2f} | pred_fes={pred_fes:.2f}', size=15)
        path = os.path.join(self.directory, f'fig25_{self.epoch}_{i}.png')
        plt.savefig(path, bbox_inches="tight")
        plt.close()


def projection(xyz, image_size=64, radius=0.025, points_per_pixel=4,
               at_vec=None, eye_vec=None, up_vec=None):

    device = xyz.device
    if at_vec is None:
        at_vec = [(0, 0, -1), (-1, 0, 0), (0, -1, 0)]
    if eye_vec is None:
        eye_vec = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
    if up_vec is None:
        up_vec = [(1, 0, 0), (0, 0, 1), (0, 0, 1)]

    gray = torch.ones(1024, 1).to(device)
    renderer = []

    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=points_per_pixel
    )
    for at, eye, up in zip(at_vec, eye_vec, up_vec):
        r, t = look_at_view_transform(at=(at,), eye=(eye,), up=(up,))
        cameras_i = FoVOrthographicCameras(device=device, R=r, T=t, znear=0.0)
        rasterizer_i = PointsRasterizer(cameras=cameras_i, raster_settings=raster_settings)

        renderer.append(PointsRenderer(
            rasterizer=rasterizer_i,
            compositor=AlphaCompositor())
                        .to(device))

    point_cloud = Pointclouds(points=[p for p in xyz], features=[gray]*len(xyz))
    xy, yz, xz = [render(point_cloud) for render in renderer]

    return [xy, yz, xz]


def projection_xz(xyz, image_size=64, radius=0.025, points_per_pixel=4,
               at_vec=None, eye_vec=None, up_vec=None):

    device = xyz.device
    if at_vec is None:
        at_vec = (0, -1, 0)
    if eye_vec is None:
        eye_vec = (0, 1, 0)
    if up_vec is None:
        up_vec = (0, 0, 1)

    gray = torch.ones(1024, 1).to(device)

    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=points_per_pixel
    )

    r, t = look_at_view_transform(at=(at_vec,), eye=(eye_vec,), up=(up_vec,))
    cameras_i = FoVOrthographicCameras(device=device, R=r, T=t, znear=0.0)
    rasterizer_i = PointsRasterizer(cameras=cameras_i, raster_settings=raster_settings)

    renderer = PointsRenderer(rasterizer=rasterizer_i,
                              compositor=AlphaCompositor()).to(device)

    point_cloud = Pointclouds(points=[p for p in xyz], features=[gray]*len(xyz))
    xz = renderer(point_cloud)

    return xz


class MyEarlyStopping(EarlyStopping):

    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer, pl_module)


def plot_2d(i, g, z_prob, potential, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    fig2, axes = plt.subplots(1, 3, num=1, figsize=(12, 4))
    axes.flatten()
    axes[0].scatter(g[:, 0], g[:, 1], s=12, alpha=0.3, c='r')
    axes[1].scatter(g[:, 1], g[:, 2], s=12, alpha=0.3, c='g')
    axes[2].scatter(g[:, 2], g[:, 0], s=12, alpha=0.3, c='b')

    fig2.text(0.3, -0.05, f'z_prob={z_prob:.3f} | potential energy={potential:.3f}', size=15)

    path = os.path.join(directory, f'xyz_{i}.png')
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_25d(i, g, z_prob, potential, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    fig25 = plt.figure(figsize=(10, 5), num=2)
    ax1 = fig25.add_subplot(111, projection='3d', elev=0, azim=45)
    ax1.scatter3D(g[:, 0], g[:, 1], g[:, 2], s=5, c='royalblue', alpha=0.5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    # ax2 = fig25.add_subplot(122, projection='3d', elev=-90, azim=-45)
    # ax2.scatter3D(g[:, 0], g[:, 1], g[:, 2], s=5, c='m', alpha=0.5)
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    fig25.text(0.25, -0.05, f'z_prob={z_prob:.3f} | potential energy={potential:.3f}', size=15)
    path = os.path.join(directory, f'fig25_{i}.png')
    plt.savefig(path, bbox_inches="tight")
    plt.close()


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, *self.shape)


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


def load_checkpoint(model, checkpoint, load_opt=False, opt=None):

    print("loading checkpoint...")
    model_dict = model.state_dict()
    model_ckpt = torch.load(checkpoint)
    pretrained_dict = model_ckpt['state_dict']
    new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
    model.load_state_dict(model_dict)
    print("loading finished!")
    if load_opt:
        opt.load_state_dict(model_ckpt['optimizer'])
        print('loaded! optimizer')
    else:
        print('not loaded optimizer')

    return model


class SaveCheckpoint(ModelCheckpoint):
    """save checkpoint after each training epoch without validation.
    if ``last_k == -1``, all models are saved. and no monitor needed in this condition.
    otherwise, please log ``global_step`` in the training_step. e.g. self.log('global_step', self.global_step)

    :param last_k: the latest k models will be saved.
    :param save_weights_only: if ``True``, only the model's weights will be saved,
    else the full model is saved.
    """
    def __init__(self, last_k=5, save_weights_only=False):
        if last_k == -1:
            super().__init__(save_top_k=-1, save_last=False, save_weights_only=save_weights_only)
        else:
            super().__init__(monitor='global_step', mode='max', save_top_k=last_k,
                             save_last=False, save_weights_only=save_weights_only)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        """
        save checkpoint after each train epoch
        """
        self.save_checkpoint(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        """
        overwrite the methods in ModelCheckpoint to avoid save checkpoint on the end of the val loop
        """
        pass
