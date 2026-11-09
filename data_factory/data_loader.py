import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class PSMSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode='train'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = pd.read_csv(data_path + '/train.csv').values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)

        test_data = pd.read_csv(data_path + '/test.csv').values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.val  = self.test

        self.test_labels = pd.read_csv(
            data_path + '/test_label.csv').values[:, 1:]

    def __len__(self):
        split = {'train': self.train, 'val': self.val,
                 'test': self.test, 'thre': self.test}[self.mode]
        denom = self.step if self.mode != 'thre' else self.win_size
        return (split.shape[0] - self.win_size) // denom + 1

    def __getitem__(self, index):
        mode = self.mode
        step = self.step if mode != 'thre' else self.win_size
        i = index * step
        split = {'train': self.train, 'val': self.val,
                 'test': self.test, 'thre': self.test}[mode]
        x = np.float32(split[i:i + self.win_size])
        y = np.float32(self.test_labels[i:i + self.win_size]
                       if mode in ('test', 'thre')
                       else self.test_labels[0:self.win_size])
        return x, y


class MSLSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode='train'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(data_path + '/MSL_train.npy')
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)
        self.val   = self.test = self.scaler.transform(
            np.load(data_path + '/MSL_test.npy'))
        self.test_labels = np.load(data_path + '/MSL_test_label.npy')

    def __len__(self):
        split = {'train': self.train, 'val': self.val,
                 'test': self.test, 'thre': self.test}[self.mode]
        denom = self.step if self.mode != 'thre' else self.win_size
        return (split.shape[0] - self.win_size) // denom + 1

    def __getitem__(self, index):
        mode = self.mode
        step = self.step if mode != 'thre' else self.win_size
        i = index * step
        split = {'train': self.train, 'val': self.val,
                 'test': self.test, 'thre': self.test}[mode]
        x = np.float32(split[i:i + self.win_size])
        y = np.float32(self.test_labels[i:i + self.win_size]
                       if mode in ('test', 'thre')
                       else self.test_labels[0:self.win_size])
        return x, y


class SMAPSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode='train'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(data_path + '/SMAP_train.npy')
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)
        self.val   = self.test = self.scaler.transform(
            np.load(data_path + '/SMAP_test.npy'))
        self.test_labels = np.load(data_path + '/SMAP_test_label.npy')

    def __len__(self):
        split = {'train': self.train, 'val': self.val,
                 'test': self.test, 'thre': self.test}[self.mode]
        denom = self.step if self.mode != 'thre' else self.win_size
        return (split.shape[0] - self.win_size) // denom + 1

    def __getitem__(self, index):
        mode = self.mode
        step = self.step if mode != 'thre' else self.win_size
        i = index * step
        split = {'train': self.train, 'val': self.val,
                 'test': self.test, 'thre': self.test}[mode]
        x = np.float32(split[i:i + self.win_size])
        y = np.float32(self.test_labels[i:i + self.win_size]
                       if mode in ('test', 'thre')
                       else self.test_labels[0:self.win_size])
        return x, y


class SMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode='train'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(data_path + '/SMD_train.npy')
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)
        n = len(self.train)
        self.val   = self.train[int(n * 0.8):]
        self.test  = self.scaler.transform(
            np.load(data_path + '/SMD_test.npy'))
        self.test_labels = np.load(data_path + '/SMD_test_label.npy')

    def __len__(self):
        split = {'train': self.train, 'val': self.val,
                 'test': self.test, 'thre': self.test}[self.mode]
        denom = self.step if self.mode != 'thre' else self.win_size
        return (split.shape[0] - self.win_size) // denom + 1

    def __getitem__(self, index):
        mode = self.mode
        step = self.step if mode != 'thre' else self.win_size
        i = index * step
        split = {'train': self.train, 'val': self.val,
                 'test': self.test, 'thre': self.test}[mode]
        x = np.float32(split[i:i + self.win_size])
        y = np.float32(self.test_labels[i:i + self.win_size]
                       if mode in ('test', 'thre')
                       else self.test_labels[0:self.win_size])
        return x, y


class SWATSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode='train'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(data_path + '/SWAT_train.npy')
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)
        n = len(self.train)
        self.val   = self.train[int(n * 0.8):]
        self.test  = self.scaler.transform(
            np.load(data_path + '/SWAT_test.npy'))
        self.test_labels = np.load(data_path + '/SWAT_test_label.npy')

    def __len__(self):
        split = {'train': self.train, 'val': self.val,
                 'test': self.test, 'thre': self.test}[self.mode]
        denom = self.step if self.mode != 'thre' else self.win_size
        return (split.shape[0] - self.win_size) // denom + 1

    def __getitem__(self, index):
        mode = self.mode
        step = self.step if mode != 'thre' else self.win_size
        i = index * step
        split = {'train': self.train, 'val': self.val,
                 'test': self.test, 'thre': self.test}[mode]
        x = np.float32(split[i:i + self.win_size])
        y = np.float32(self.test_labels[i:i + self.win_size]
                       if mode in ('test', 'thre')
                       else self.test_labels[0:self.win_size])
        return x, y


def get_loader_segment(data_path, batch_size, win_size=100, step=1,
                       mode='train', dataset='PSM'):
    loaders = {
        'PSM':  PSMSegLoader,
        'MSL':  MSLSegLoader,
        'SMAP': SMAPSegLoader,
        'SMD':  SMDSegLoader,
        'SWAT': SWATSegLoader,
    }
    if dataset not in loaders:
        raise ValueError(f"未知数据集：{dataset}，支持：{list(loaders)}")

    ds = loaders[dataset](data_path, win_size, step, mode)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=(mode == 'train'), num_workers=0)
