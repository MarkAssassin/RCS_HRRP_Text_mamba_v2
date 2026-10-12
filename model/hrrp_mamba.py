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


# Mamba网络分类
class Mamba_Net(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, use_fc=True):
        super(Mamba_Net, self).__init__()
        self.mamba_layer = Mamba( # 输出与输入维度相同，且特征维度不需要整除头数
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=d_model, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
            )
        self.use_fc = use_fc
        # MLP
        self.fc = nn.Sequential(

            nn.Linear(d_model*d_model, 2048),  # 将接好的特征向量映射到4
            nn.Mish(),
            nn.Linear(2048, 512),
            nn.Mish(),
            nn.Linear(512, 128)
        )
    
    def forward(self, hrrp):
        output = self.mamba_layer(hrrp)
        if self.use_fc:
            output = output.view(output.shape[0], -1)
            output = self.fc(output)
            output = torch.squeeze(output, dim=1)
        return output

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    hrrp = torch.zeros((64, 101, 101)).to(device)
    model = Mamba_Net(d_model=101, use_fc=True)
    model.to(device)
    hrrp_feat = model(hrrp) # (64, 128)
    print(hrrp_feat.shape)