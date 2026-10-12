import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import TensorDataset  # TensorDataset可以用来对tensor数据进行打包,该类中的 tensor 第一维度必须相等(即每一个图片对应一个标签)
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import scipy.io
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from mamba_ssm import Mamba

from model.cnn import rcs_cnn
from model.tcn import TemporalConvNet
from model.couple_mamba import MultiMamba
# from cnn import rcs_cnn
# from tcn import TemporalConvNet
# from couple_mamba import MultiMamba

Tensor = torch.Tensor

def positional_encoding(X, num_features, dropout_p=0.0, max_len=512) -> Tensor: #这里的max_len不用改为101，因为512>101，如果维度大于512了再改
    r'''
        给输入加入位置编码 X对应d_in, num_featrues对应d_hid, 即num_features=d_hid=d_model=100
    参数：
        - num_features: 输入进来的维度
        - dropout_p: dropout的概率，当其为非零元素时执行dropout
        - max_len: 句子的最大长度，默认512

    形状：
        - 输入： [batch_size, seq_length, num_features]
        - 输出： [batch_size, seq_length, num_features]
    例子：
        >>> X = torch.randn((2,4,10))
        >>> X = positional_encoding(X, 10)
        >>> print(X.shape)
        >>> torch.Size([2, 4, 10])
    '''

    dropout = nn.Dropout(dropout_p)
    P = torch.zeros((1, max_len, num_features))  # 行
    X_ = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
        10000,
        torch.arange(0, num_features, 2,
                     dtype=torch.float32) / num_features)  # 2i/d_model    num_features -> d_model  i维度指d_model章的维度
    P[:, :, 0::2] = torch.sin(X_)  # 偶数列用sin 奇数列用cos
    P[:, :, 1::2] = torch.cos(X_)
    X = X + P[:, :X.shape[1], :].to(X.device)  # X是已经embedding P复制batchsize份 加到X上
    return dropout(X)

class Time_Freq_Net_Coupled(nn.Module):
    def __init__(self, d_model=128, split_num=4, temperature=0.5):
        super(Time_Freq_Net_Coupled, self).__init__()
        num_channels = [16,32,64]
        self.d_model = d_model
        self.d_feature = int(d_model/split_num)
        self.temperature = temperature
        self.cnn = rcs_cnn(feature_dim=d_model) # Freq
        self.tcn = TemporalConvNet(num_inputs=1, num_channels=num_channels, kernel_size=2, dropout=0.2, use_fc=True) # Time
        self.mamba_layer = MultiMamba( # 输出与输入维度相同，且特征维度不需要整除头数
            d_model=self.d_feature, # Model dimension d_model d_model是句子长度, token是句子个数
            n_layer=2,
            use_dw_1=False, # gamma
            use_dw_2=True # alpha, beta 
            )
    
    def forward(self, rcs_time, rcs_fft):
        rcs_time, rcs_fft = self.tcn(rcs_time), self.cnn(rcs_fft) # (64, 128)
        # rcs_time, rcs_fft = rcs_time.unsqueeze(1), rcs_fft.unsqueeze(1) # (64, 1, 128)
        rcs_time, rcs_fft = rcs_time.reshape(-1, 4, 32), rcs_fft.reshape(-1, 4, 32) # (64, 4, 32)
        # rcs_time = rcs_time + positional_encoding(rcs_time, self.d_model) # (64, 1, 128)
        # rcs_fft = rcs_fft + positional_encoding(rcs_fft, self.d_model)
        rcs_time = rcs_time + positional_encoding(rcs_time, self.d_feature) # (64, 4, 32)
        rcs_fft = rcs_fft + positional_encoding(rcs_fft, self.d_feature)
        rcs_fusion = self.mamba_layer(rcs_time, rcs_fft) # (64, 128)
        return rcs_fusion

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    rcs_time = torch.zeros((64, 1, 101)).to(device=device)
    rcs_freq = torch.zeros((64, 1, 101)).to(device=device)
    model = Time_Freq_Net_Coupled()
    model.to(device)
    rcs_fusion = model(rcs_time, rcs_freq) # (64, 128)
    print(rcs_fusion.shape)