import os
import cv2
import numpy as np
import torch
from numpy.random import RandomState
from torch.utils.data import Dataset
import h5py
import glob


class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.dataset = name
        self.mat_files = open(self.dataset, 'r').readlines()
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num * 100

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        gt_file = file_name.split(' ')[1][:-1]
        img_file = file_name.split(' ')[0]

        O = cv2.imread(img_file)
        B = cv2.imread(gt_file)
        M = np.clip((O - B).sum(axis=2), 0, 1).astype(np.float32)
        O = np.transpose(O.astype(np.float32) / 255, (2, 0, 1))
        B = np.transpose(B.astype(np.float32) / 255, (2, 0, 1))

        sample = {'O': O, 'B': B, 'M': M}

        return sample


class TrainDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.dataset = name
        self.h5f = h5py.File(self.dataset, 'r')
        self.keys = list(self.h5f.keys())
        assert len(self.keys) % 3 == 0
        self.len = int(len(self.keys) / 3)
        print('total {} samples \n'.format(self.len))
        self.h5f.close()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        self.h5f = h5py.File(self.dataset, 'r')
        O = self.h5f['{}_O'.format(index)]
        B = self.h5f['{}_B'.format(index)]
        M = self.h5f['{}_M'.format(index)]
        O, B, M = torch.Tensor(np.array(O)), torch.Tensor(
            np.array(B)), torch.Tensor(np.array(M))
        # print(index, O.shape, B.shape, M.shape)
        sample = {'O': O, 'B': B, 'M': M}
        self.h5f.close()
        return sample


class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = name
        self.mat_files = open(self.root_dir, 'r').readlines()

        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]

        gt_file = "." + file_name.split(' ')[1][:-1]
        img_file = "." + file_name.split(' ')[0]

        O = cv2.imread(img_file).astype(np.float32) / 255.0
        B = cv2.imread(gt_file).astype(np.float32) / 255.0

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))

        sample = {'O': O, 'B': B, 'M': O}

        return sample


class TestDataset2(Dataset):
    def __init__(self, name):
        super(TestDataset2, self).__init__()
        self.dataset = name
        self.h5f = h5py.File(self.dataset, 'r')
        self.keys = list(self.h5f.keys())
        assert len(self.keys) % 3 == 0
        self.len = int(len(self.keys) / 3)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        O = self.h5f['{}_O'.format(index)]
        B = self.h5f['{}_B'.format(index)]
        M = self.h5f['{}_M'.format(index)]
        O, B, M = torch.Tensor(np.array(O)), torch.Tensor(
            np.array(B)), torch.Tensor(np.array(M))
        sample = {'O': O, 'B': B, 'M': M}
        return sample


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j +
                        1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def generate_train_data(data_path, patch_size, stride):
    h5f = h5py.File(os.path.join(data_path, 'train.h5'))
    file_nums = len(glob.glob(os.path.join(data_path, 'data', '*.png')))
    print('total files {}'.format(file_nums))
    index = 0
    for i in range(file_nums):
        img_name = os.path.join(data_path, 'data', '{}_rain.png'.format(i))
        gt_name = os.path.join(data_path, 'gt', '{}_clean.png'.format(i))
        O = cv2.imread(img_name)
        B = cv2.imread(gt_name)
        M = np.clip((O - B).sum(axis=2), 0, 1).astype(np.float32)
        M = M[np.newaxis, :, :]
        O = np.transpose(O.astype(np.float32) / 255, (2, 0, 1))
        B = np.transpose(B.astype(np.float32) / 255, (2, 0, 1))
        print(O.shape, B.shape, M.shape)
        O_patches, B_patches, M_patches = Im2Patch(
            O, patch_size, stride), Im2Patch(B, patch_size, stride), Im2Patch(
                M, patch_size, stride)
        for j in range(O_patches.shape[-1]):
            h5f.create_dataset(
                name='{}_O'.format(index), data=O_patches[:, :, :, j])
            h5f.create_dataset(
                name='{}_B'.format(index), data=B_patches[:, :, :, j])
            h5f.create_dataset(
                name='{}_M'.format(index), data=M_patches[0, :, :, j])
            index += 1
        print('img {}, total samples {}'.format(i, index))


def generate_test_data(data_path):
    h5f = h5py.File(os.path.join(data_path, 'test.h5'))
    file_nums = len(glob.glob(os.path.join(data_path, 'data', '*.png')))
    print('total files {}'.format(file_nums))
    index = 0
    for i in range(file_nums):
        img_name = os.path.join(data_path, 'data', '{}_rain.png'.format(i))
        gt_name = os.path.join(data_path, 'gt', '{}_clean.png'.format(i))
        O = cv2.imread(img_name)
        B = cv2.imread(gt_name)
        M = np.clip((O - B).sum(axis=2), 0, 1).astype(np.float32)
        O = np.transpose(O.astype(np.float32) / 255, (2, 0, 1))
        B = np.transpose(B.astype(np.float32) / 255, (2, 0, 1))
        h5f.create_dataset(name='{}_O'.format(index), data=O)
        h5f.create_dataset(name='{}_B'.format(index), data=B)
        h5f.create_dataset(name='{}_M'.format(index), data=M)
        index += 1
        print('img {}, total samples {}'.format(i, index))


if __name__ == "__main__":
    data_path = '/data0/niejiangtao/ICIP2019Deraining/test_a'
    # generate_train_data(data_path, 100, 80)
    # generate_test_data(data_path)