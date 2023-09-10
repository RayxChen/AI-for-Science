import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
from pytorch_lightning.callbacks import EarlyStopping


def get_data(batch_size=150, data_dir='./10K'):

    xyz = np.load(f'{data_dir}/xyz.npz')['arr_0']
    y = np.load(f'{data_dir}/FES_01.npz')['arr_0']
    xrd = np.load(f'{data_dir}/xrd_60_100.npz')['arr_0']
    # T = np.array(np.arange(10, 500, 490/10000)).astype('float32')[:, None]
    T = np.load(f'{data_dir}/T_01.npz')['arr_0']

    # class Data(Dataset):
    #     def __init__(self):
    #         self.xyz = xyz
    #         self.y = y
    #         self.T = T
    #
    #     def __getitem__(self, index):
    #         return self.X[index], self.y[index], self.T[index]

    class Data(Dataset):
        def __init__(self):
            self.xyz = xyz
            self.xrd = xrd
            self.y = y
            self.T = T

        def __getitem__(self, index):
            return self.xyz[index], self.xrd[index], self.y[index], self.T[index]

        def __len__(self):
            return len(self.y)

    full_dataset = Data()

    train_size = int(0.75 * len(full_dataset))
    valid_size = int(0.25 * train_size)
    test_size = len(full_dataset) - train_size

    train_data, test_data = random_split(full_dataset, [train_size, test_size])
    train_data, valid_data = random_split(train_data, [train_size - valid_size, valid_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=20, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=20, pin_memory=True)

    return train_loader, valid_loader, test_loader


class MyEarlyStopping(EarlyStopping):

    def on_validation_end(self, trainer, pl_module):
        if pl_module.current_epoch > 25:
            self._run_early_stopping_check(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        pass


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
