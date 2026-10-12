import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import TensorDataset  # TensorDataset可以用来对tensor数据进行打包,该类中的 tensor 第一维度必须相等(即每一个图片对应一个标签)
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import scipy.io
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from utils.encoder_utils import TextEncoder
# from encoder_utils import TextEncoder

class Dataset:
    def __init__(self,
                 data_path:str,
                 use_gpu = True,
                 batch_size:int = 256):
        self.data_path = data_path
        self.use_gpu = use_gpu
        self.batch_size = batch_size

    def load_data_mat(self):
        data = scipy.io.loadmat(self.data_path)
        train_data = abs(data['train_data'])
        train_label = abs(data['train_labels'])
        test_data = abs(data['test_data'])
        test_label = abs(data['test_labels'])
        self.train_list = [train_data, train_label]
        self.test_list = [test_data, test_label]
        return self.train_list, self.test_list

    def create_dataloader(self, type='train'):
        # 设置随机种子
        torch.manual_seed(seed=1000)
        g = torch.Generator()
        if type == 'train':
            [data,label], _ = self.load_data_mat()
            shuffle = True
            generator = g
            if 'HRRP' in self.data_path:
                data = np.transpose(data, (2, 0, 1))  # 先将维度变换为(7260,101,101)
                # data = np.expand_dims(data, axis=1)  # 再将维度变换为(3630,1,101,101)
            elif 'RCS' in self.data_path:
                data = np.expand_dims(data, axis=1) # #将维度变换为(4840,1,101) 
            self.data_mean_train, self.data_std_train = data.mean(), data.std()
        elif type == 'test':
            _, [data, label] = self.load_data_mat()
            shuffle = False
            generator = None
            if 'HRRP' in self.data_path:
                data = np.transpose(data, (2, 0, 1))  # 先将维度变换为(7260,101,101)
                # data = np.expand_dims(data, axis=1)  # 再将维度变换为(3630,1,101,101)
            elif 'RCS' in self.data_path:
                data = np.expand_dims(data, axis=1) # #将维度变换为(4840,1,101) 
        data, label = map(torch.tensor, (data, label.squeeze()))
        # 数据集标准化
        data = (data-self.data_mean_train)/self.data_std_train
        # 计算均值和标准差
        # mean = data.mean()
        # std = data.std()
        # normalize = transforms.Normalize(mean=mean, std=std)
        # data = normalize(data)
        # print(data.shape) # torch.Size([4840, 1, 14])

        # if self.use_gpu:
        #     data, label = data.cuda(), label.cuda()
        tensor_dataset = TensorDataset(data, label)  # 相当于将图片和对应的标签打包在了一起
        data_loader = DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=shuffle, generator=generator)
        return data_loader

    def get_train_label(self):
        [data, label], _ = self.load_data_mat()
        return label

    def get_test_label(self):
        _, [data, label] = self.load_data_mat()
        return label

class Dataset_Text:
    def __init__(self,
                 data_path:str,
                 use_gpu = True,
                 batch_size:int = 256):
        self.data_path = data_path
        self.use_gpu = use_gpu
        self.batch_size = batch_size

    def load_data_mat(self):
        text_data = list(np.load(self.data_path))
        return text_data

    def create_dataloader(self, type='train'):
        train_num = 4840 # 4840个训练样本
        test_num = 1452 # 1452个测试样本
        # 获得文本编码
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        text_data = self.load_data_mat()
        text_encoder = TextEncoder(device)
        text_feature = text_encoder(text_data).cpu().numpy()
        # 设置随机种子
        torch.manual_seed(seed=1000)
        g = torch.Generator()
        if type == 'train':
            text_feature = text_feature[:train_num]
            text_label = np.zeros((1,train_num)) # 4840个训练样本, 后续仅使用文本特征,标签未使用
            shuffle = True
            generator = g
        elif type == 'test':
            text_feature = text_feature[train_num:]
            text_label = np.zeros((1,test_num)) # 1452个测试样本
            shuffle = False
            generator = None
        text_feature, text_label = map(torch.tensor, (text_feature, text_label.squeeze()))
        tensor_dataset = TensorDataset(text_feature, text_label)  # 相当于将图片和对应的标签打包在了一起
        data_loader = DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=shuffle, generator=generator)
        return data_loader


if __name__ == '__main__':
    # data_type = 'HRRP'  # 'RCS' or 'HRRP'
    # dataset = Dataset(data_path=f'dataset/{data_type}_data_X_20dB.mat', batch_size=256)
    # train_list, test_list = dataset.load_data_mat()
    # [train_data, train_label] = train_list
    # [test_data, test_label] = test_list
    # print(train_data.shape) # (4840, 101) or (101, 101, 4840)
    # print(test_data.shape) # (1452, 101) or (101, 101, 1452)
    # data_loader = dataset.create_dataloader()

    dataset = Dataset_Text(data_path='text_creator/L_band_text_data.npy', batch_size=256)
    data_loader = dataset.create_dataloader()

