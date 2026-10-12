import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch.nn.parameter import Parameter
from torch.nn.init import constant_, xavier_normal_
from torch.nn.init import xavier_uniform_
import warnings

'''
    用于提取rcs频谱特征
'''
class rcs_cnn(nn.Module):  # n=101   (n+2p-f)/s+1
    def __init__(self, feature_dim):
        super(rcs_cnn, self).__init__()
        self.cnn = nn.Sequential(
            # Layer1
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),  # n=101
            nn.BatchNorm1d(8),
            nn.Mish(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 50

            # Layer2
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),  # n=50
            nn.BatchNorm1d(16),
            nn.Mish(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 25

            # Layer3
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),  # n=25
            nn.BatchNorm1d(32),
            nn.Mish(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 12

            # Layer4
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),  # n=25
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 6

            )

        self.fc = nn.Linear(64*6, feature_dim, bias=False)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)  # 矩阵的每一行就是这个批量中每张图片的各个参数，即矩阵中一行对应一张图片
        x = self.fc(x)
        return x

# hrrp特征
class hrrp_cnn(nn.Module):  # n=101   (n+2p-f)/s+1
    def __init__(self, feature_dim):
        super(hrrp_cnn, self).__init__()
        self.cnn = nn.Sequential(
            # Layer1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),  # n=101
            nn.BatchNorm2d(8),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 50

            # Layer2
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),  # n=50
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 25

            # Layer3
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),  # n=25
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 12

            # Layer4
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),  # n=25
            nn.BatchNorm2d(64),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 6

            )

        self.fc = nn.Linear(64*6*6, feature_dim, bias=False)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)  # 矩阵的每一行就是这个批量中每张图片的各个参数，即矩阵中一行对应一张图片
        x = self.fc(x)
        return x
# import pytest
if __name__ == '__main__':
    hrrp = torch.zeros((64, 1, 101, 101))
    model = hrrp_cnn(feature_dim=128)
    feat = model(hrrp)
    print(feat.shape)  # torch.Size([64, 128])


