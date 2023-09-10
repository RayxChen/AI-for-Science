import torch
import numpy as np
import matplotlib.pyplot as plt


def calXRD(xyz, qhist=60, theta_min=10, theta_max=40, lamda=1.5406, maxR=8.8, atom='Na'):
    """Na maxR=8.8, Al maxR=9
       Na 35.2, Al 21.507036"""

    device = xyz.device
    xyz = xyz.transpose(1, 2)  # B, C, N -> B, N, C
    B, N, C = xyz.size()
    xyz = 35.2 * (xyz + 1) / 2
    theta_step = (theta_max - theta_min) / qhist
    theta = torch.arange(theta_min, theta_max, theta_step).view(1, qhist, 1).to(device)
    pi = 3.141592

    q = 4 * pi * torch.sin(theta * pi / 180) / lamda
    
    if atom == 'Na':
        fij = 4.7626 * torch.exp(-3.2850 * (q / (4 * pi)) * (q / (4 * pi))) + \
                 3.1736 * torch.exp(-8.8422 * (q / (4 * pi)) * (q / (4 * pi))) + \
                 1.2674 * torch.exp(-0.3136 * (q / (4 * pi)) * (q / (4 * pi))) + \
                 1.1128 * torch.exp(-129.424 * (q / (4 * pi)) * (q / (4 * pi))) + 0.676
    
    elif atom == 'Al':
        fij = 6.4202 * torch.exp(-3.0387 * (q / (4 * pi)) * (q / (4 * pi))) + \
                 1.9002 * torch.exp(-0.7426 * (q / (4 * pi)) * (q / (4 * pi))) + \
                 1.5936 * torch.exp(-31.5472 * (q / (4 * pi)) * (q / (4 * pi))) + \
                 1.9646 * torch.exp(-85.0886 * (q / (4 * pi)) * (q / (4 * pi))) + 1.1151
    
    elif atom == 'Mg':
        fij = 5.4204 * torch.exp(-2.8275 * (q / (4 * pi)) * (q / (4 * pi))) + \
                 2.1735 * torch.exp(-79.2611 * (q / (4 * pi)) * (q / (4 * pi))) + \
                 1.2269 * torch.exp(-0.3808 * (q / (4 * pi)) * (q / (4 * pi))) + \
                 2.3073 * torch.exp(-7.1937 * (q / (4 * pi)) * (q / (4 * pi))) + 0.8584
    else:
        raise ValueError()

    R_ij = torch.cdist(xyz, xyz)
    R_ij = R_ij[:, torch.triu(torch.ones(N, N), diagonal=1) == 1].flatten(start_dim=1, end_dim=-1)
    R_ij = torch.where(R_ij < maxR, R_ij, torch.zeros_like(R_ij))

    R_ij = R_ij.reshape(B, 1, -1)

    v = torch.where(R_ij != 0,
                    torch.sin(q * R_ij) / (q * R_ij) * torch.sin(pi * R_ij / maxR) / (pi * R_ij / maxR) / N,
                    R_ij)

    v = torch.einsum('bqi->bq', v)
    fij = fij.squeeze()
    intensity = fij * fij * (2*v + 1)

    return intensity  # , 2 * theta.squeeze()


XYZ = np.load(f'../data/50K/xyz.npz', allow_pickle=True)['arr_0']
# model = calXRD().cuda()

batch_size = 50
n = 1000
data = []
for i in range(n):
    xyz = XYZ[i * batch_size:(i + 1) * batch_size]
    xyz = torch.tensor(xyz).cuda()

    intensity = calXRD(xyz)
    data.append(intensity.cpu().numpy())

    torch.cuda.empty_cache()

data = np.concatenate(data, 0)

np.savez_compressed('xrd.npz', data)


# xyz = np.load(f'../data_10K/xyz.npz', allow_pickle=True)['arr_0'][:3]
#
# xyz = torch.tensor(xyz).cuda()
#
# intensity, two_theta = calXRD1(xyz)
#
# for i in range(len(xyz)):
#     plt.scatter(two_theta.cpu().numpy(), intensity[i].cpu().numpy())
#     plt.savefig(f'{i+8}.png')
#     plt.close()