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
from torch.nn.utils.parametrizations import weight_norm
from mamba_ssm import Mamba

from model.rcs_tf_fusion import positional_encoding
from model.tcn import TemporalBlock, TemporalConvNet
from model.cnn import rcs_cnn
from model.couple_mamba import MultiMamba
from model.hrrp_mamba import Mamba_Net
from model.rht_fusion import RHTAlign_Net
from model.gat import GATFusion3_v3, GATFusion3_v3_nopAll, GATFusion_singleInput, GATFusion_2input
from model.attention import TransformerEncoderLayer_Double, TransformerEncoderLayer_Treble
from model.models import RHT_Net

# from rcs_tf_fusion import positional_encoding
# from tcn import TemporalBlock, TemporalConvNet
# from cnn import rcs_cnn
# from couple_mamba import MultiMamba
# from rht_fusion import RHTAlign_Net
# from gat import GATFusion3_v3, GATFusion3_v3_nopAll, GATFusion_singleInput, GATFusion_2input
# from attention import TransformerEncoderLayer_Double, TransformerEncoderLayer_Treble
# from models import RHT_Net

# rcs时域
class TemporalConvNet_ablation(nn.Module):
    def __init__(self, num_inputs=1, num_channels=[16,32,64], kernel_size=2, dropout=0.2, use_fc=True, output_dim=4):
        super(TemporalConvNet_ablation, self).__init__()
        self.use_fc = use_fc
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i # 1 2 4
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
 
        self.network = nn.Sequential(*layers)

        self.fc = nn.Sequential(
        nn.Linear(101*num_channels[-1], 1024),
        nn.Mish(),
        nn.Linear(1024, 512),
        nn.Mish(),
        nn.Linear(512, 128),
        nn.Mish(),
        nn.Linear(128, output_dim),
        )
 
    def forward(self, x):
        x = self.network(x)
        if self.use_fc:
            x = x.view(x.shape[0], -1)
            x = self.fc(x)
        return x

# rcs频域
class rcs_cnn_ablation(nn.Module):  # n=101   (n+2p-f)/s+1
    def __init__(self, output_dim):
        super(rcs_cnn_ablation, self).__init__()
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

        self.fc = nn.Sequential(
            nn.Linear(64*6, 128, bias=False),
            nn.Mish(),
            nn.Linear(128, output_dim, bias=False)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)  # 矩阵的每一行就是这个批量中每张图片的各个参数，即矩阵中一行对应一张图片
        x = self.fc(x)
        return x

# rcs时域频域融合
class Time_Freq_Net_Coupled(nn.Module):
    def __init__(self, d_model=128, output_dim=4, split_num=4, temperature=0.5, use_cl=True, use_rec=False):
        super(Time_Freq_Net_Coupled, self).__init__()
        num_channels = [16,32,64]
        self.d_model = d_model
        self.d_feature = int(d_model/split_num)
        self.temperature = temperature
        self.use_cl = use_cl
        self.use_rec = use_rec
        self.cnn = rcs_cnn(feature_dim=d_model) # Freq
        self.tcn = TemporalConvNet(num_inputs=1, num_channels=num_channels, kernel_size=2, dropout=0.2, use_fc=True) # Time
        self.mamba_layer = MultiMamba( # 输出与输入维度相同，且特征维度不需要整除头数
            d_model=self.d_feature, # Model dimension d_model d_model是句子长度, token是句子个数
            n_layer=2,
            use_dw_1=False, # gamma
            use_dw_2=True # alpha, beta
            )
        
        self.fc = nn.Sequential(
            nn.Linear(64, 16),
            nn.Mish(),
            nn.Linear(16, output_dim),
            nn.Mish()
        )
    
    def forward(self, rcs_time, rcs_fft):
        rcs_time, rcs_fft = self.tcn(rcs_time), self.cnn(rcs_fft) # (64, 128)
        rcs_time, rcs_fft = rcs_time.reshape(-1, 4, 32), rcs_fft.reshape(-1, 4, 32) # (64, 4, 32)
        rcs_time = rcs_time + positional_encoding(rcs_time, self.d_feature) # (64, 4, 32)
        rcs_fft = rcs_fft + positional_encoding(rcs_fft, self.d_feature)
        rcs_fusion = self.mamba_layer(rcs_time, rcs_fft) # (64, 64)
        if self.use_rec:
            output = self.fc(rcs_fusion)
        else:
            output = rcs_fusion
        return output

# hrrp
# Mamba网络分类
class Mamba_Net_ablation(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, use_fc=True, output_dim=4):
        super(Mamba_Net_ablation, self).__init__()
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
            nn.Linear(512, output_dim)
        )
    
    def forward(self, hrrp):
        output = self.mamba_layer(hrrp)
        if self.use_fc:
            output = output.view(output.shape[0], -1)
            output = self.fc(output)
            output = torch.squeeze(output, dim=1)
        return output

# rcs+hrrp+text
class GATFusion3_Net_ablation(nn.Module):
    def __init__(self, rht_output_dim, gat_hidden_dim, gat_output_dim, use_cl=True, feat_type='L', num_class=4):
        super().__init__()

        # 三个频段的特征抽取器（你可自行换成 GRU/CNN）
        # L/S/X频段rcs&hrrp&text融合特征
        self.use_cl = use_cl
        self.freq_feature = RHT_Net(use_cl=use_cl) # (batch, gru_output_dim)
        # L/S/X频段图注意力融合模块
        self.gat_fusion = GATFusion3_v3_nopAll(input_dim=rht_output_dim, hidden_dim=gat_hidden_dim, output_dim=gat_output_dim, feat_type=feat_type)

        self.fc = nn.Sequential(
            nn.Linear(gat_output_dim, 64),
            nn.Mish(),
            nn.Linear(64, num_class)
        )

    def forward(self, rcs_L_time, rcs_L_fft, hrrp_L, text_L, rcs_S_time, rcs_S_fft, hrrp_S, text_S, rcs_X_time, rcs_X_fft, hrrp_X, text_X):
        if self.use_cl:
            L_feature_list, cl_loss_L = self.freq_feature(rcs_L_time, rcs_L_fft, hrrp_L, text_L)
            S_feature_list, cl_loss_S = self.freq_feature(rcs_S_time, rcs_S_fft, hrrp_S, text_S)
            X_feature_list, cl_loss_X = self.freq_feature(rcs_X_time, rcs_X_fft, hrrp_X, text_X)
            cl_loss = 0.3*cl_loss_L + 0.3*cl_loss_S + 0.4*cl_loss_X
        else:
            L_feature_list, S_feature_list, X_feature_list = self.freq_feature(rcs_L_time, rcs_L_fft, hrrp_L, text_L), self.freq_feature(rcs_S_time, rcs_S_fft, hrrp_S, text_S), self.freq_feature(rcs_X_time, rcs_X_fft, hrrp_X, text_X)
        [rcs_feat_L, hrrp_feat_L, text_feat_L] = L_feature_list # (64, 128)
        [rcs_feat_S, hrrp_feat_S, text_feat_S] = S_feature_list
        [rcs_feat_X, hrrp_feat_X, text_feat_X] = X_feature_list

        fusion_feature = self.gat_fusion(rcs_feat_L, hrrp_feat_L, text_feat_L,
                                         rcs_feat_S, hrrp_feat_S, text_feat_S,
                                         rcs_feat_X, hrrp_feat_X, text_feat_X)
        output = self.fc(fusion_feature)

        if self.use_cl:
            return output, cl_loss
        else:
            return output

################################## 多频段单模态 ##################################
# rcs融合特征
class Multi_RCS_Coupled(nn.Module):
    def __init__(self, d_model=128, output_dim=4, split_num=4):
        super(Multi_RCS_Coupled, self).__init__()
        num_channels = [16,32,64]
        self.d_model = d_model
        self.d_feature = int(d_model/split_num)
        self.cnn = rcs_cnn(feature_dim=d_model) # Freq
        self.tcn = TemporalConvNet(num_inputs=1, num_channels=num_channels, kernel_size=2, dropout=0.2, use_fc=True) # Time
        self.mamba_layer = MultiMamba( # 输出与输入维度相同，且特征维度不需要整除头数
            d_model=self.d_feature, # Model dimension d_model d_model是句子长度, token是句子个数
            n_layer=2,
            use_dw_1=False, # gamma
            use_dw_2=True # alpha, beta
            )
        
        self.fc = nn.Sequential(
            nn.Linear(64, 16),
            nn.Mish(),
            nn.Linear(16, output_dim),
            nn.Mish()
        )
    
    def forward(self, rcs_L_time, rcs_L_fft, rcs_S_time, rcs_S_fft, rcs_X_time, rcs_X_fft):
        # Process each frequency band
        rcs_L_time, rcs_L_fft = self.tcn(rcs_L_time), self.cnn(rcs_L_fft)
        rcs_S_time, rcs_S_fft = self.tcn(rcs_S_time), self.cnn(rcs_S_fft)
        rcs_X_time, rcs_X_fft = self.tcn(rcs_X_time), self.cnn(rcs_X_fft)

        # Reshape and add positional encoding
        rcs_L_time, rcs_L_fft = rcs_L_time.reshape(-1, 4, 32), rcs_L_fft.reshape(-1, 4, 32)
        rcs_S_time, rcs_S_fft = rcs_S_time.reshape(-1, 4, 32), rcs_S_fft.reshape(-1, 4, 32)
        rcs_X_time, rcs_X_fft = rcs_X_time.reshape(-1, 4, 32), rcs_X_fft.reshape(-1, 4, 32)

        rcs_L_time = rcs_L_time + positional_encoding(rcs_L_time, self.d_feature)
        rcs_L_fft = rcs_L_fft + positional_encoding(rcs_L_fft, self.d_feature)
        rcs_S_time = rcs_S_time + positional_encoding(rcs_S_time, self.d_feature)
        rcs_S_fft = rcs_S_fft + positional_encoding(rcs_S_fft, self.d_feature)
        rcs_X_time = rcs_X_time + positional_encoding(rcs_X_time, self.d_feature)
        rcs_X_fft = rcs_X_fft + positional_encoding(rcs_X_fft, self.d_feature)

        # Apply Mamba layer
        rcs_L_fusion = self.mamba_layer(rcs_L_time, rcs_L_fft)
        rcs_S_fusion = self.mamba_layer(rcs_S_time, rcs_S_fft)
        rcs_X_fusion = self.mamba_layer(rcs_X_time, rcs_X_fft)
        output = [rcs_L_fusion, rcs_S_fusion, rcs_X_fusion]
        return output

class Multi_RCS_single_ablation(nn.Module):
    def __init__(self, gat_input_dim, gat_hidden_dim, gat_output_dim, num_class=4):
        super().__init__()
        self.multi_net = Multi_RCS_Coupled()
        self.gat_fusion = GATFusion_singleInput(input_dim=gat_input_dim, hidden_dim=gat_hidden_dim, output_dim=gat_output_dim)
        self.fc = nn.Sequential(
            nn.Linear(gat_output_dim, 64),
            nn.Mish(),
            nn.Linear(64, num_class)
        )
    
    def forward(self, rcs_L_time, rcs_L_fft, rcs_S_time, rcs_S_fft, rcs_X_time, rcs_X_fft):
        multi_feature = self.multi_net(rcs_L_time, rcs_L_fft, rcs_S_time, rcs_S_fft, rcs_X_time, rcs_X_fft)
        gat_feature = self.gat_fusion(multi_feature[0], multi_feature[1], multi_feature[2])
        output = self.fc(gat_feature)
        return output
    

# hrrp特征
class Multi_HRRP(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super(Multi_HRRP, self).__init__()
        self.mamba_layer = Mamba( # 输出与输入维度相同，且特征维度不需要整除头数
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=d_model, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
            )
        # MLP
        self.fc = nn.Sequential(

            nn.Linear(d_model*d_model, 2048),  # 将接好的特征向量映射到4
            nn.Mish(),
            nn.Linear(2048, 512),
            nn.Mish(),
            nn.Linear(512, 128)
        )
    
    def forward(self, hrrp_L, hrrp_S, hrrp_X):
        output_L = self.mamba_layer(hrrp_L)
        output_S = self.mamba_layer(hrrp_S)
        output_X = self.mamba_layer(hrrp_X)
        output_L = output_L.view(output_L.shape[0], -1)
        output_L = self.fc(output_L)
        output_S = output_S.view(output_S.shape[0], -1)
        output_S = self.fc(output_S)
        output_X = output_X.view(output_X.shape[0], -1)
        output_X = self.fc(output_X)
        output = [output_L, output_S, output_X]
        return output

class Multi_HRRP_single_ablation(nn.Module):
    def __init__(self, gat_input_dim, gat_hidden_dim, gat_output_dim, d_model=101, num_class=4):
        super().__init__()
        self.multi_net = Multi_HRRP(d_model=d_model)
        self.gat_fusion = GATFusion_singleInput(input_dim=gat_input_dim, hidden_dim=gat_hidden_dim, output_dim=gat_output_dim)
        self.fc = nn.Sequential(
            nn.Linear(gat_output_dim, 64),
            nn.Mish(),
            nn.Linear(64, num_class)
        )
    
    def forward(self, hrrp_L, hrrp_S, hrrp_X):
        multi_feature = self.multi_net(hrrp_L, hrrp_S, hrrp_X)
        gat_feature = self.gat_fusion(multi_feature[0], multi_feature[1], multi_feature[2])
        output = self.fc(gat_feature)
        return output

################################## 多频段双模态 ##################################
# RCS+文本
class RTAlign_Net(nn.Module):
    def __init__(self, temperature=0.5, use_cl=True):
        super(RTAlign_Net, self).__init__()
        self.use_cl = use_cl
        self.temperature = temperature
        self.multi_net = Multi_RCS_Coupled()

    def contrastive_loss(self, x, y):
        logits = x @ y.T / self.temperature # Similar Matrix
        labels = torch.arange(x.size(0), device=x.device) # 这里不应用真实的标签，为了让相似度矩阵接近单位阵，需要设置one-hot标签 1 0 0
                                                                                                                             #   0 1 0
                                                                                                                             #   0 0 1
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2

    def forward(self, rcs_L_time, rcs_L_fft, text_L, rcs_S_time, rcs_S_fft, text_S, rcs_X_time, rcs_X_fft, text_X):
        rcs_fusion = self.multi_net(rcs_L_time, rcs_L_fft, rcs_S_time, rcs_S_fft, rcs_X_time, rcs_X_fft) # (64, 128)
        rcs_L_fusion, rcs_S_fusion, rcs_X_fusion = rcs_fusion[0], rcs_fusion[1], rcs_fusion[2]
        rcs_L_norm, rcs_S_norm, rcs_X_norm = F.normalize(rcs_L_fusion, p=2, dim=1), F.normalize(rcs_S_fusion, p=2, dim=1), F.normalize(rcs_X_fusion, p=2, dim=1)
        text_L_norm, text_S_norm, text_X_norm = F.normalize(text_L, p=2, dim=1), F.normalize(text_S, p=2, dim=1), F.normalize(text_X, p=2, dim=1)
        cl_L, cl_S, cl_X = self.contrastive_loss(rcs_L_norm, text_L_norm), self.contrastive_loss(rcs_S_norm, text_S_norm), self.contrastive_loss(rcs_X_norm, text_X_norm)
        feat_list_L = [rcs_L_fusion, text_L]
        feat_list_S = [rcs_S_fusion, text_S]
        feat_list_X = [rcs_X_fusion, text_X]
        cl_loss = 0.3*cl_L + 0.3*cl_S + 0.4*cl_X
        if self.use_cl:
            return [feat_list_L, feat_list_S, feat_list_X], cl_loss
        else:
            return [feat_list_L, feat_list_S, feat_list_X]

class Multi_RCS_Text_double_ablation(nn.Module):
    def __init__(self, gat_input_dim, gat_hidden_dim, gat_output_dim, use_cl=True, num_class=4):
        super().__init__()
        self.use_cl = use_cl
        self.multi_net = RTAlign_Net()
        self.gat_fusion = GATFusion_2input(input_dim=gat_input_dim, hidden_dim=gat_hidden_dim, output_dim=gat_output_dim)
        self.fc = nn.Sequential(
            nn.Linear(gat_output_dim, 64),
            nn.Mish(),
            nn.Linear(64, num_class)
        )
    
    def forward(self, rcs_L_time, rcs_L_fft, text_L, rcs_S_time, rcs_S_fft, text_S, rcs_X_time, rcs_X_fft, text_X):
        feat_list, cl_loss = self.multi_net(rcs_L_time, rcs_L_fft, text_L, rcs_S_time, rcs_S_fft, text_S, rcs_X_time, rcs_X_fft, text_X)
        rcs_L_fusion, text_L = feat_list[0][0], feat_list[0][1]
        rcs_S_fusion, text_S = feat_list[1][0], feat_list[1][1]
        rcs_X_fusion, text_X = feat_list[2][0], feat_list[2][1]

        gat_feature = self.gat_fusion(rcs_L_fusion, text_L, rcs_S_fusion, text_S, rcs_X_fusion, text_X)
        output = self.fc(gat_feature)
        return output, cl_loss

# RCS+HRRP
class RHAlign_Net(nn.Module):
    def __init__(self, d_model=101, temperature=0.5, use_cl=True):
        super(RHAlign_Net, self).__init__()
        self.use_cl = use_cl
        self.temperature = temperature
        self.multi_net = Multi_RCS_Coupled()
        self.multi_hrrp = Multi_HRRP(d_model=d_model)

    def contrastive_loss(self, x, y):
        logits = x @ y.T / self.temperature # Similar Matrix
        labels = torch.arange(x.size(0), device=x.device) # 这里不应用真实的标签，为了让相似度矩阵接近单位阵，需要设置one-hot标签 1 0 0
                                                                                                                             #   0 1 0
                                                                                                                             #   0 0 1
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2

    def forward(self, rcs_L_time, rcs_L_fft, hrrp_L, rcs_S_time, rcs_S_fft, hrrp_S, rcs_X_time, rcs_X_fft, hrrp_X):
        rcs_fusion = self.multi_net(rcs_L_time, rcs_L_fft, rcs_S_time, rcs_S_fft, rcs_X_time, rcs_X_fft) # (64, 128)
        rcs_L_fusion, rcs_S_fusion, rcs_X_fusion = rcs_fusion[0], rcs_fusion[1], rcs_fusion[2]
        hrrp_fusion = self.multi_hrrp(hrrp_L, hrrp_S, hrrp_X)
        hrrp_L, hrrp_S, hrrp_X = hrrp_fusion[0], hrrp_fusion[1], hrrp_fusion[2]
        rcs_L_norm, rcs_S_norm, rcs_X_norm = F.normalize(rcs_L_fusion, p=2, dim=1), F.normalize(rcs_S_fusion, p=2, dim=1), F.normalize(rcs_X_fusion, p=2, dim=1)
        hrrp_L_norm, hrrp_S_norm, hrrp_X_norm = F.normalize(hrrp_L, p=2, dim=1), F.normalize(hrrp_S, p=2, dim=1), F.normalize(hrrp_X, p=2, dim=1)
        cl_L, cl_S, cl_X = self.contrastive_loss(rcs_L_norm, hrrp_L_norm), self.contrastive_loss(rcs_S_norm, hrrp_S_norm), self.contrastive_loss(rcs_X_norm, hrrp_X_norm)
        feat_list_L = [rcs_L_fusion, hrrp_L]
        feat_list_S = [rcs_S_fusion, hrrp_S]
        feat_list_X = [rcs_X_fusion, hrrp_X]
        cl_loss = 0.3*cl_L + 0.3*cl_S + 0.4*cl_X
        if self.use_cl:
            return [feat_list_L, feat_list_S, feat_list_X], cl_loss
        else:
            return [feat_list_L, feat_list_S, feat_list_X]

class Multi_RCS_HRRP_double_ablation(nn.Module):
    def __init__(self, gat_input_dim, gat_hidden_dim, gat_output_dim, use_cl=True, num_class=4):
        super().__init__()
        self.use_cl = use_cl
        self.multi_net = RHAlign_Net()
        self.gat_fusion = GATFusion_2input(input_dim=gat_input_dim, hidden_dim=gat_hidden_dim, output_dim=gat_output_dim)
        self.fc = nn.Sequential(
            nn.Linear(gat_output_dim, 64),
            nn.Mish(),
            nn.Linear(64, num_class)
        )
    
    def forward(self, rcs_L_time, rcs_L_fft, hrrp_L, rcs_S_time, rcs_S_fft, hrrp_S, rcs_X_time, rcs_X_fft, hrrp_X):
        feat_list, cl_loss = self.multi_net(rcs_L_time, rcs_L_fft, hrrp_L, rcs_S_time, rcs_S_fft, hrrp_S, rcs_X_time, rcs_X_fft, hrrp_X)
        rcs_L_fusion, hrrp_L = feat_list[0][0], feat_list[0][1]
        rcs_S_fusion, hrrp_S = feat_list[1][0], feat_list[1][1]
        rcs_X_fusion, hrrp_X = feat_list[2][0], feat_list[2][1]

        gat_feature = self.gat_fusion(rcs_L_fusion, hrrp_L, rcs_S_fusion, hrrp_S, rcs_X_fusion, hrrp_X)
        output = self.fc(gat_feature)
        return output, cl_loss

# hrrp+text
class HTAlign_Net(nn.Module):
    def __init__(self, d_model=101, temperature=0.5, use_cl=True):
        super(HTAlign_Net, self).__init__()
        self.use_cl = use_cl
        self.temperature = temperature
        self.multi_net = Multi_HRRP(d_model=d_model)

    def contrastive_loss(self, x, y):
        logits = x @ y.T / self.temperature # Similar Matrix
        labels = torch.arange(x.size(0), device=x.device) # 这里不应用真实的标签，为了让相似度矩阵接近单位阵，需要设置one-hot标签 1 0 0
                                                                                                                             #   0 1 0
                                                                                                                             #   0 0 1
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2

    def forward(self, hrrp_L, text_L, hrrp_S, text_S, hrrp_X, text_X):
        hrrp_fusion = self.multi_net(hrrp_L, hrrp_S, hrrp_X)
        hrrp_L, hrrp_S, hrrp_X = hrrp_fusion[0], hrrp_fusion[1], hrrp_fusion[2]
        text_L_norm, text_S_norm, text_X_norm = F.normalize(text_L, p=2, dim=1), F.normalize(text_S, p=2, dim=1), F.normalize(text_X, p=2, dim=1)
        hrrp_L_norm, hrrp_S_norm, hrrp_X_norm = F.normalize(hrrp_L, p=2, dim=1), F.normalize(hrrp_S, p=2, dim=1), F.normalize(hrrp_X, p=2, dim=1)
        cl_L, cl_S, cl_X = self.contrastive_loss(text_L_norm, hrrp_L_norm), self.contrastive_loss(text_S_norm, hrrp_S_norm), self.contrastive_loss(text_X_norm, hrrp_X_norm)
        feat_list_L = [hrrp_L, text_L]
        feat_list_S = [hrrp_S, text_S]
        feat_list_X = [hrrp_X, text_X]
        cl_loss = 0.3*cl_L + 0.3*cl_S + 0.4*cl_X
        if self.use_cl:
            return [feat_list_L, feat_list_S, feat_list_X], cl_loss
        else:
            return [feat_list_L, feat_list_S, feat_list_X]

class Multi_HRRP_Text_double_ablation(nn.Module):
    def __init__(self, gat_input_dim, gat_hidden_dim, gat_output_dim, use_cl=True, num_class=4):
        super().__init__()
        self.use_cl = use_cl
        self.multi_net = HTAlign_Net()
        self.gat_fusion = GATFusion_2input(input_dim=gat_input_dim, hidden_dim=gat_hidden_dim, output_dim=gat_output_dim)
        self.fc = nn.Sequential(
            nn.Linear(gat_output_dim, 64),
            nn.Mish(),
            nn.Linear(64, num_class)
        )
    
    def forward(self, hrrp_L, text_L, hrrp_S, text_S, hrrp_X, text_X):
        feat_list, cl_loss = self.multi_net(hrrp_L, text_L, hrrp_S, text_S, hrrp_X, text_X)
        hrrp_L, text_L = feat_list[0][0], feat_list[0][1]
        hrrp_S, text_S = feat_list[1][0], feat_list[1][1]
        hrrp_X, text_X = feat_list[2][0], feat_list[2][1]

        gat_feature = self.gat_fusion(hrrp_L, text_L, hrrp_S, text_S, hrrp_X, text_X)
        output = self.fc(gat_feature)
        return output, cl_loss


################################## 多频段单模态_Baseline ##################################
# Baseline表示融合都用cross-attention进行融合
# rcs融合特征
class Multi_RCS_Coupled_Baseline(nn.Module):
    def __init__(self, d_model=128, split_num=4):
        super(Multi_RCS_Coupled_Baseline, self).__init__()
        num_channels = [16,32,64]
        self.d_model = d_model
        self.d_feature = int(d_model/split_num)
        self.cnn = rcs_cnn(feature_dim=d_model) # Freq
        self.tcn = TemporalConvNet(num_inputs=1, num_channels=num_channels, kernel_size=2, dropout=0.2, use_fc=True) # Time
        self.attention = TransformerEncoderLayer_Double(d_model=self.d_feature, nhead=4)
    
    def forward(self, rcs_L_time, rcs_L_fft, rcs_S_time, rcs_S_fft, rcs_X_time, rcs_X_fft):
        # Process each frequency band
        rcs_L_time, rcs_L_fft = self.tcn(rcs_L_time), self.cnn(rcs_L_fft)
        rcs_S_time, rcs_S_fft = self.tcn(rcs_S_time), self.cnn(rcs_S_fft)
        rcs_X_time, rcs_X_fft = self.tcn(rcs_X_time), self.cnn(rcs_X_fft)

        # Reshape and add positional encoding
        rcs_L_time, rcs_L_fft = rcs_L_time.reshape(-1, 4, 32), rcs_L_fft.reshape(-1, 4, 32)
        rcs_S_time, rcs_S_fft = rcs_S_time.reshape(-1, 4, 32), rcs_S_fft.reshape(-1, 4, 32)
        rcs_X_time, rcs_X_fft = rcs_X_time.reshape(-1, 4, 32), rcs_X_fft.reshape(-1, 4, 32)

        rcs_L_time = rcs_L_time + positional_encoding(rcs_L_time, self.d_feature)
        rcs_L_fft = rcs_L_fft + positional_encoding(rcs_L_fft, self.d_feature)
        rcs_S_time = rcs_S_time + positional_encoding(rcs_S_time, self.d_feature)
        rcs_S_fft = rcs_S_fft + positional_encoding(rcs_S_fft, self.d_feature)
        rcs_X_time = rcs_X_time + positional_encoding(rcs_X_time, self.d_feature)
        rcs_X_fft = rcs_X_fft + positional_encoding(rcs_X_fft, self.d_feature)

        # Apply Mamba layer
        rcs_L_fusion = self.attention(rcs_L_time, rcs_L_fft)
        rcs_L_fusion = rcs_L_fusion.reshape(-1, 128)
        rcs_S_fusion = self.attention(rcs_S_time, rcs_S_fft)
        rcs_S_fusion = rcs_S_fusion.reshape(-1, 128)
        rcs_X_fusion = self.attention(rcs_X_time, rcs_X_fft)
        rcs_X_fusion = rcs_X_fusion.reshape(-1, 128)

        output = [rcs_L_fusion, rcs_S_fusion, rcs_X_fusion]
        return output

class Single_RCS_single_ablation_baseline(nn.Module):
    def __init__(self, input_dim=128, d_model=128, split_num=4, nhead=4, num_class=4):
        super().__init__()
        num_channels = [16,32,64]
        self.d_model = d_model
        self.d_feature = int(d_model/split_num)
        self.tcn = TemporalConvNet(num_inputs=1, num_channels=num_channels, kernel_size=2, dropout=0.2, use_fc=True) # Time
        self.attention = TransformerEncoderLayer_Treble(d_model=int(input_dim/split_num), nhead=nhead)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, num_class)
        )
    
    def forward(self, rcs_L_time, rcs_S_time, rcs_X_time):
        rcs_L_time, rcs_S_time, rcs_X_time = self.tcn(rcs_L_time), self.tcn(rcs_S_time), self.tcn(rcs_X_time)
        rcs_L_time, rcs_S_time, rcs_X_time= rcs_L_time.reshape(-1, 4, 32), rcs_S_time.reshape(-1, 4, 32), rcs_X_time.reshape(-1, 4, 32)
        rcs_L_time = rcs_L_time + positional_encoding(rcs_L_time, self.d_feature)
        rcs_S_time = rcs_S_time + positional_encoding(rcs_S_time, self.d_feature)
        rcs_X_time = rcs_X_time + positional_encoding(rcs_X_time, self.d_feature)
        fusion_feat = self.attention(rcs_L_time, rcs_S_time, rcs_X_time)
        fusion_feat = fusion_feat.reshape(-1, 128)
        output = self.fc(fusion_feat)
        return output

class Multi_RCS_single_ablation_baseline(nn.Module):
    def __init__(self, input_dim=128, split_num=4, nhead=4, num_class=4):
        super().__init__()
        self.multi_net = Multi_RCS_Coupled_Baseline()
        self.attention = TransformerEncoderLayer_Treble(d_model=int(input_dim/split_num), nhead=nhead)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, num_class)
        )
    
    def forward(self, rcs_L_time, rcs_L_fft, rcs_S_time, rcs_S_fft, rcs_X_time, rcs_X_fft):
        multi_feature = self.multi_net(rcs_L_time, rcs_L_fft, rcs_S_time, rcs_S_fft, rcs_X_time, rcs_X_fft)
        rcs_L_fusion, rcs_S_fusion , rcs_X_fusion = multi_feature[0].reshape(-1, 4, 32), multi_feature[1].reshape(-1, 4, 32), multi_feature[2].reshape(-1, 4, 32)
        fusion_feat = self.attention(rcs_L_fusion, rcs_S_fusion, rcs_X_fusion)
        fusion_feat = fusion_feat.reshape(-1, 128)
        output = self.fc(fusion_feat)
        return output

# hrrp特征
class Multi_HRRP_single_ablation_baseline(nn.Module):
    def __init__(self, d_model=101, input_dim=128, split_num=4, nhead=4, num_class=4):
        super().__init__()
        self.multi_net = Multi_HRRP(d_model=d_model)
        self.attention = TransformerEncoderLayer_Treble(d_model=int(input_dim/split_num), nhead=nhead)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, num_class)
        )
    
    def forward(self, hrrp_L, hrrp_S, hrrp_X):
        multi_feature = self.multi_net(hrrp_L, hrrp_S, hrrp_X)
        hrrp_L_fusion, hrrp_S_fusion , hrrp_X_fusion = multi_feature[0].reshape(-1, 4, 32), multi_feature[1].reshape(-1, 4, 32), multi_feature[2].reshape(-1, 4, 32)
        fusion_feat = self.attention(hrrp_L_fusion, hrrp_S_fusion, hrrp_X_fusion)
        fusion_feat = fusion_feat.reshape(-1, 128)
        output = self.fc(fusion_feat)
        return output

################################## 多频段双模态(Baseline) ##################################
# RCS+文本
class RTAlign_Net_Baseline(nn.Module):
    def __init__(self, d_model=128*2, split_num=4, temperature=0.5, use_cl=True, num_class=4):
        super(RTAlign_Net_Baseline, self).__init__()
        self.use_cl = use_cl
        self.d_feature = int(d_model/split_num)
        self.temperature = temperature
        self.multi_net = Multi_RCS_Coupled_Baseline()
        # self.attention1 = TransformerEncoderLayer_Double(d_model=self.d_feature, nhead=4)
        self.attention2 = TransformerEncoderLayer_Treble(d_model=self.d_feature, nhead=4)
        self.fc = nn.Sequential(
            nn.Linear(128*2, 64),
            nn.Mish(),
            nn.Linear(64, num_class)
        )

    def contrastive_loss(self, x, y):
        logits = x @ y.T / self.temperature # Similar Matrix
        labels = torch.arange(x.size(0), device=x.device) # 这里不应用真实的标签，为了让相似度矩阵接近单位阵，需要设置one-hot标签 1 0 0
                                                                                                                             #   0 1 0
                                                                                                                             #   0 0 1
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2

    def forward(self, rcs_L_time, rcs_L_fft, text_L, rcs_S_time, rcs_S_fft, text_S, rcs_X_time, rcs_X_fft, text_X):
        rcs_fusion = self.multi_net(rcs_L_time, rcs_L_fft, rcs_S_time, rcs_S_fft, rcs_X_time, rcs_X_fft) # (64, 128)
        rcs_L_fusion, rcs_S_fusion, rcs_X_fusion = rcs_fusion[0], rcs_fusion[1], rcs_fusion[2]
        rcs_L_norm, rcs_S_norm, rcs_X_norm = F.normalize(rcs_L_fusion, p=2, dim=1), F.normalize(rcs_S_fusion, p=2, dim=1), F.normalize(rcs_X_fusion, p=2, dim=1)
        text_L_norm, text_S_norm, text_X_norm = F.normalize(text_L, p=2, dim=1), F.normalize(text_S, p=2, dim=1), F.normalize(text_X, p=2, dim=1)
        cl_L, cl_S, cl_X = self.contrastive_loss(rcs_L_norm, text_L_norm), self.contrastive_loss(rcs_S_norm, text_S_norm), self.contrastive_loss(rcs_X_norm, text_X_norm)
        # rcs_L_fusion, text_L = rcs_L_fusion.reshape(-1, 4, 32), text_L.reshape(-1, 4, 32)
        # rcs_S_fusion, text_S = rcs_S_fusion.reshape(-1, 4, 32), text_S.reshape(-1, 4, 32)
        # rcs_X_fusion, text_X = rcs_X_fusion.reshape(-1, 4, 32), text_X.reshape(-1, 4, 32)
        # L_feat_fusion, S_feat_fusion, X_feat_fusion = self.attention1(rcs_L_fusion, text_L), self.attention1(rcs_S_fusion, text_S), self.attention1(rcs_X_fusion, text_X)
        fusion_feat_L = torch.cat([rcs_L_fusion, text_L], dim=1)
        fusion_feat_S = torch.cat([rcs_S_fusion, text_S], dim=1)
        fusion_feat_X = torch.cat([rcs_X_fusion, text_X], dim=1)
        fusion_feat_L, fusion_feat_S, fusion_feat_X = fusion_feat_L.reshape(-1, 4, 64), fusion_feat_S.reshape(-1, 4, 64), fusion_feat_X.reshape(-1, 4, 64)
        fusion_feat = self.attention2(fusion_feat_L, fusion_feat_S, fusion_feat_X)
        fusion_feat = fusion_feat.reshape(-1, 128*2)
        cl_loss = 0.3*cl_L + 0.3*cl_S + 0.4*cl_X
        output = self.fc(fusion_feat)
        if self.use_cl:
            return output, cl_loss
        else:
            return output

# RCS+HRRP
class RHAlign_Net_Baseline(nn.Module):
    def __init__(self, d_model=101, input_dim=128*2, split_num=4, temperature=0.5, use_cl=True, num_class=4):
        super(RHAlign_Net_Baseline, self).__init__()
        self.use_cl = use_cl
        self.d_feature = int(input_dim/split_num)
        self.temperature = temperature
        self.multi_net = Multi_RCS_Coupled_Baseline()
        self.multi_hrrp = Multi_HRRP(d_model=d_model)
        self.attention1 = TransformerEncoderLayer_Double(d_model=self.d_feature, nhead=4)
        self.attention2 = TransformerEncoderLayer_Treble(d_model=self.d_feature, nhead=4)
        self.fc = nn.Sequential(
            nn.Linear(128*2, 64),
            nn.Mish(),
            nn.Linear(64, num_class)
        )

    def contrastive_loss(self, x, y):
        logits = x @ y.T / self.temperature # Similar Matrix
        labels = torch.arange(x.size(0), device=x.device) # 这里不应用真实的标签，为了让相似度矩阵接近单位阵，需要设置one-hot标签 1 0 0
                                                                                                                             #   0 1 0
                                                                                                                             #   0 0 1
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2

    def forward(self, rcs_L_time, rcs_L_fft, hrrp_L, rcs_S_time, rcs_S_fft, hrrp_S, rcs_X_time, rcs_X_fft, hrrp_X):
        rcs_fusion = self.multi_net(rcs_L_time, rcs_L_fft, rcs_S_time, rcs_S_fft, rcs_X_time, rcs_X_fft) # (64, 128)
        rcs_L_fusion, rcs_S_fusion, rcs_X_fusion = rcs_fusion[0], rcs_fusion[1], rcs_fusion[2]
        hrrp_fusion = self.multi_hrrp(hrrp_L, hrrp_S, hrrp_X)
        hrrp_L, hrrp_S, hrrp_X = hrrp_fusion[0], hrrp_fusion[1], hrrp_fusion[2]
        rcs_L_norm, rcs_S_norm, rcs_X_norm = F.normalize(rcs_L_fusion, p=2, dim=1), F.normalize(rcs_S_fusion, p=2, dim=1), F.normalize(rcs_X_fusion, p=2, dim=1)
        hrrp_L_norm, hrrp_S_norm, hrrp_X_norm = F.normalize(hrrp_L, p=2, dim=1), F.normalize(hrrp_S, p=2, dim=1), F.normalize(hrrp_X, p=2, dim=1)
        cl_L, cl_S, cl_X = self.contrastive_loss(rcs_L_norm, hrrp_L_norm), self.contrastive_loss(rcs_S_norm, hrrp_S_norm), self.contrastive_loss(rcs_X_norm, hrrp_X_norm)
        # rcs_L_fusion, hrrp_L = rcs_L_fusion.reshape(-1, 4, 32), hrrp_L.reshape(-1, 4, 32)
        # rcs_S_fusion, hrrp_S = rcs_S_fusion.reshape(-1, 4, 32), hrrp_S.reshape(-1, 4, 32)
        # rcs_X_fusion, hrrp_X = rcs_X_fusion.reshape(-1, 4, 32), hrrp_X.reshape(-1, 4, 32)
        # L_feat_fusion, S_feat_fusion, X_feat_fusion = self.attention1(rcs_L_fusion, hrrp_L), self.attention1(rcs_S_fusion, hrrp_S), self.attention1(rcs_X_fusion, hrrp_X)
        fusion_feat_L = torch.cat([rcs_L_fusion, hrrp_L], dim=1)
        fusion_feat_S = torch.cat([rcs_S_fusion, hrrp_S], dim=1)
        fusion_feat_X = torch.cat([rcs_X_fusion, hrrp_X], dim=1)
        fusion_feat_L, fusion_feat_S, fusion_feat_X = fusion_feat_L.reshape(-1, 4, 64), fusion_feat_S.reshape(-1, 4, 64), fusion_feat_X.reshape(-1, 4, 64)
        fusion_feat = self.attention2(fusion_feat_L, fusion_feat_S, fusion_feat_X)
        fusion_feat = fusion_feat.reshape(-1, 128*2)
        cl_loss = 0.3*cl_L + 0.3*cl_S + 0.4*cl_X
        output = self.fc(fusion_feat)
        if self.use_cl:
            return output, cl_loss
        else:
            return output

# HRRP+文本
class HTAlign_Net_Baseline(nn.Module):
    def __init__(self, d_model=101, input_dim=128*2, split_num=4, temperature=0.5, use_cl=True, num_class=4):
        super(HTAlign_Net_Baseline, self).__init__()
        self.use_cl = use_cl
        self.d_feature = int(input_dim/split_num)
        self.temperature = temperature
        self.multi_hrrp = Multi_HRRP(d_model=d_model)
        self.attention1 = TransformerEncoderLayer_Double(d_model=self.d_feature, nhead=4)
        self.attention2 = TransformerEncoderLayer_Treble(d_model=self.d_feature, nhead=4)
        self.fc = nn.Sequential(
            nn.Linear(128*2, 64),
            nn.Mish(),
            nn.Linear(64, num_class)
        )

    def contrastive_loss(self, x, y):
        logits = x @ y.T / self.temperature # Similar Matrix
        labels = torch.arange(x.size(0), device=x.device) # 这里不应用真实的标签，为了让相似度矩阵接近单位阵，需要设置one-hot标签 1 0 0
                                                                                                                             #   0 1 0
                                                                                                                             #   0 0 1
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2

    def forward(self, hrrp_L, text_L, hrrp_S, text_S, hrrp_X, text_X):
        hrrp_fusion = self.multi_hrrp(hrrp_L, hrrp_S, hrrp_X)
        hrrp_L, hrrp_S, hrrp_X = hrrp_fusion[0], hrrp_fusion[1], hrrp_fusion[2]
        hrrp_L_norm, hrrp_S_norm, hrrp_X_norm = F.normalize(hrrp_L, p=2, dim=1), F.normalize(hrrp_S, p=2, dim=1), F.normalize(hrrp_X, p=2, dim=1)
        text_L_norm, text_S_norm, text_X_norm = F.normalize(text_L, p=2, dim=1), F.normalize(text_S, p=2, dim=1), F.normalize(text_X, p=2, dim=1)
        cl_L, cl_S, cl_X = self.contrastive_loss(hrrp_L_norm, text_L_norm), self.contrastive_loss(hrrp_S_norm, text_S_norm), self.contrastive_loss(hrrp_X_norm, text_X_norm)
        # hrrp_L_fusion, text_L = hrrp_L.reshape(-1, 4, 32), text_L.reshape(-1, 4, 32)
        # hrrp_S_fusion, text_S = hrrp_S.reshape(-1, 4, 32), text_S.reshape(-1, 4, 32)
        # hrrp_X_fusion, text_X = hrrp_X.reshape(-1, 4, 32), text_X.reshape(-1, 4, 32)
        # L_feat_fusion, S_feat_fusion, X_feat_fusion = self.attention1(hrrp_L_fusion, text_L), self.attention1(hrrp_S_fusion, text_S), self.attention1(hrrp_X_fusion, text_X)
        fusion_feat_L = torch.cat([hrrp_L, text_L], dim=1)
        fusion_feat_S = torch.cat([hrrp_S, text_S], dim=1)
        fusion_feat_X = torch.cat([hrrp_X, text_X], dim=1)
        fusion_feat_L, fusion_feat_S, fusion_feat_X = fusion_feat_L.reshape(-1, 4, 64), fusion_feat_S.reshape(-1, 4, 64), fusion_feat_X.reshape(-1, 4, 64)
        fusion_feat = self.attention2(fusion_feat_L, fusion_feat_S, fusion_feat_X)
        fusion_feat = fusion_feat.reshape(-1, 128*2)
        cl_loss = 0.3*cl_L + 0.3*cl_S + 0.4*cl_X
        output = self.fc(fusion_feat)
        if self.use_cl:
            return output, cl_loss
        else:
            return output


################################## 各模块有效性 ##################################
# TFDF
class RHTAlign_Net_TFDF(nn.Module):
    def __init__(self, d_model=101, temperature=0.5, use_cl=True):
        super(RHTAlign_Net_TFDF, self).__init__()
        self.use_cl = use_cl
        self.temperature = temperature
        self.rcs_fusion = Time_Freq_Net_Coupled()
        self.hrrp_mamba = Mamba_Net(d_model=d_model, use_fc=True)

    def contrastive_loss(self, x, y):
        logits = x @ y.T / self.temperature # Similar Matrix
        labels = torch.arange(x.size(0), device=x.device) # 这里不应用真实的标签，为了让相似度矩阵接近单位阵，需要设置one-hot标签 1 0 0
                                                                                                                             #   0 1 0
                                                                                                                             #   0 0 1
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2

    def forward(self, rcs_time, rcs_fft, hrrp, text):
        rcs_feature = self.rcs_fusion(rcs_time, rcs_fft) # (64, 128) 
        hrrp_feature = self.hrrp_mamba(hrrp) # (64, 128)
        text_feature = text  # text 已经是 (B, feature_dim) 形式
        # rcs_gru = self.gru(rcs_feature.unsqueeze(1)) # (batch, gru_output_dim)
        # hrrp_gru = self.gru(hrrp_feature.unsqueeze(1)) # (batch, gru_output_dim)
        
        rcs_norm = F.normalize(rcs_feature, p=2, dim=1)
        hrrp_norm = F.normalize(hrrp_feature, p=2, dim=1)
        text_norm = F.normalize(text_feature, p=2, dim=1)
        # hrrp&text对齐
        contrast_loss_1 = self.contrastive_loss(hrrp_norm, text_norm)
        # rcs&text对齐  
        contrast_loss_2 = self.contrastive_loss(rcs_norm, text_norm)
        contrast_loss = 0.5 * (contrast_loss_1 + contrast_loss_2)
        # 特征序列
        feat_list = [rcs_feature, hrrp_feature, text_feature]
        if self.use_cl:
            return feat_list, contrast_loss
        else:
            return feat_list

class RHT_Net_TFDF(nn.Module):
    def __init__(self, use_cl=True):
        super(RHT_Net_TFDF, self).__init__()
        self.use_cl = use_cl
        self.fusion_net = RHTAlign_Net_TFDF(use_cl=use_cl)
        
    def forward(self, rcs_time, rcs_fft, hrrp, text):
        if self.use_cl:
            feature_list, cl_loss = self.fusion_net(rcs_time, rcs_fft, hrrp, text)
            return feature_list, cl_loss
        else:
            feature_list = self.fusion_net(rcs_time, rcs_fft, hrrp, text)
            return feature_list

class GATFusion3_Net_TFDF(nn.Module):
    def __init__(self, input_dim=128, split_num=4, nhead=4, use_cl=True, num_class=4):
        super().__init__()

        # 三个频段的特征抽取器（你可自行换成 GRU/CNN）
        # L/S/X频段rcs&hrrp&text融合特征
        self.use_cl = use_cl
        self.freq_feature = RHT_Net_TFDF(use_cl=use_cl) # (batch, gru_output_dim)
        self.attention = TransformerEncoderLayer_Treble(d_model=int(input_dim/split_num), nhead=nhead)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, num_class)
        )

    def forward(self, rcs_L_time, rcs_L_fft, hrrp_L, text_L, rcs_S_time, rcs_S_fft, hrrp_S, text_S, rcs_X_time, rcs_X_fft, hrrp_X, text_X):
        if self.use_cl:
            L_feature_list, cl_loss_L = self.freq_feature(rcs_L_time, rcs_L_fft, hrrp_L, text_L)
            S_feature_list, cl_loss_S = self.freq_feature(rcs_S_time, rcs_S_fft, hrrp_S, text_S)
            X_feature_list, cl_loss_X = self.freq_feature(rcs_X_time, rcs_X_fft, hrrp_X, text_X)
            cl_loss = 0.3*cl_loss_L + 0.3*cl_loss_S + 0.4*cl_loss_X
        else:
            L_feature_list, S_feature_list, X_feature_list = self.freq_feature(rcs_L_time, rcs_L_fft, hrrp_L, text_L), self.freq_feature(rcs_S_time, rcs_S_fft, hrrp_S, text_S), self.freq_feature(rcs_X_time, rcs_X_fft, hrrp_X, text_X)
        [rcs_feat_L, hrrp_feat_L, text_feat_L] = L_feature_list # (64, 128)
        [rcs_feat_S, hrrp_feat_S, text_feat_S] = S_feature_list
        [rcs_feat_X, hrrp_feat_X, text_feat_X] = X_feature_list

        rcs_feat_L, hrrp_feat_L, text_feat_L = rcs_feat_L.reshape(-1, 4, 32), hrrp_feat_L.reshape(-1, 4, 32), text_feat_L.reshape(-1, 4, 32)
        rcs_feat_S, hrrp_feat_S, text_feat_S = rcs_feat_S.reshape(-1, 4, 32), hrrp_feat_S.reshape(-1, 4, 32), text_feat_S.reshape(-1, 4, 32)
        rcs_feat_X, hrrp_feat_X, text_feat_X = rcs_feat_X.reshape(-1, 4, 32), hrrp_feat_X.reshape(-1, 4, 32), text_feat_X.reshape(-1, 4, 32)
        fusion_feat_L = self.attention(rcs_feat_L, hrrp_feat_L, text_feat_L)
        fusion_feat_S = self.attention(rcs_feat_S, hrrp_feat_S, text_feat_S)
        fusion_feat_X = self.attention(rcs_feat_X, hrrp_feat_X, text_feat_X)
        fusion_feat = self.attention(fusion_feat_L, fusion_feat_S, fusion_feat_X)
        fusion_feat = fusion_feat.reshape(-1, 128)
        output = self.fc(fusion_feat)

        if self.use_cl:
            return output, cl_loss
        else:
            return output

# MGAT
class Time_Freq_Net_Coupled_MGAT(nn.Module):
    def __init__(self, d_model=128, split_num=4, temperature=0.5):
        super(Time_Freq_Net_Coupled_MGAT, self).__init__()
        num_channels = [16,32,64]
        self.d_model = d_model
        self.d_feature = int(d_model/split_num)
        self.temperature = temperature
        self.cnn = rcs_cnn(feature_dim=d_model) # Freq
        self.tcn = TemporalConvNet(num_inputs=1, num_channels=num_channels, kernel_size=2, dropout=0.2, use_fc=True) # Time
        self.attention = TransformerEncoderLayer_Double(d_model=self.d_feature, nhead=4)
    
    def forward(self, rcs_time, rcs_fft):
        rcs_time, rcs_fft = self.tcn(rcs_time), self.cnn(rcs_fft) # (64, 128)
        rcs_time, rcs_fft = rcs_time.reshape(-1, 4, 32), rcs_fft.reshape(-1, 4, 32)
        rcs_fusion = self.attention(rcs_time, rcs_fft)
        rcs_fusion = rcs_fusion.reshape(-1, 128)
        return rcs_fusion

class RHTAlign_Net_MGAT(nn.Module):
    def __init__(self, d_model=101, temperature=0.5, use_cl=True):
        super(RHTAlign_Net_MGAT, self).__init__()
        self.use_cl = use_cl
        self.temperature = temperature
        self.rcs_fusion = Time_Freq_Net_Coupled_MGAT()
        self.hrrp_mamba = Mamba_Net(d_model=d_model, use_fc=True)

    def contrastive_loss(self, x, y):
        logits = x @ y.T / self.temperature # Similar Matrix
        labels = torch.arange(x.size(0), device=x.device) # 这里不应用真实的标签，为了让相似度矩阵接近单位阵，需要设置one-hot标签 1 0 0
                                                                                                                             #   0 1 0
                                                                                                                             #   0 0 1
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2

    def forward(self, rcs_time, rcs_fft, hrrp, text):
        rcs_feature = self.rcs_fusion(rcs_time, rcs_fft) # (64, 128) 
        hrrp_feature = self.hrrp_mamba(hrrp) # (64, 128)
        text_feature = text  # text 已经是 (B, feature_dim) 形式
        # rcs_gru = self.gru(rcs_feature.unsqueeze(1)) # (batch, gru_output_dim)
        # hrrp_gru = self.gru(hrrp_feature.unsqueeze(1)) # (batch, gru_output_dim)
        
        rcs_norm = F.normalize(rcs_feature, p=2, dim=1)
        hrrp_norm = F.normalize(hrrp_feature, p=2, dim=1)
        text_norm = F.normalize(text_feature, p=2, dim=1)
        # hrrp&text对齐
        contrast_loss_1 = self.contrastive_loss(hrrp_norm, text_norm)
        # rcs&text对齐  
        contrast_loss_2 = self.contrastive_loss(rcs_norm, text_norm)
        contrast_loss = 0.5 * (contrast_loss_1 + contrast_loss_2)
        # 特征序列
        feat_list = [rcs_feature, hrrp_feature, text_feature]
        if self.use_cl:
            return feat_list, contrast_loss
        else:
            return feat_list

class RHT_Net_MGAT(nn.Module):
    def __init__(self, use_cl=True):
        super(RHT_Net_MGAT, self).__init__()
        self.use_cl = use_cl
        self.fusion_net = RHTAlign_Net_MGAT(use_cl=use_cl)
        
    def forward(self, rcs_time, rcs_fft, hrrp, text):
        if self.use_cl:
            feature_list, cl_loss = self.fusion_net(rcs_time, rcs_fft, hrrp, text)
            return feature_list, cl_loss
        else:
            feature_list = self.fusion_net(rcs_time, rcs_fft, hrrp, text)
            return feature_list

class GATFusion3_Net_MGAT(nn.Module):
    def __init__(self, rht_output_dim, gat_hidden_dim, gat_output_dim, use_cl=True, num_class=4):
        super().__init__()

        # 三个频段的特征抽取器（你可自行换成 GRU/CNN）
        # L/S/X频段rcs&hrrp&text融合特征
        self.use_cl = use_cl
        self.freq_feature = RHT_Net_MGAT(use_cl=use_cl) # (batch, gru_output_dim)
        # L/S/X频段图注意力融合模块
        self.gat_fusion = GATFusion3_v3_nopAll(input_dim=rht_output_dim, hidden_dim=gat_hidden_dim, output_dim=gat_output_dim)

        self.fc = nn.Sequential(
            nn.Linear(gat_output_dim, 64),
            nn.Mish(),
            nn.Linear(64, num_class)
        )

    def forward(self, rcs_L_time, rcs_L_fft, hrrp_L, text_L, rcs_S_time, rcs_S_fft, hrrp_S, text_S, rcs_X_time, rcs_X_fft, hrrp_X, text_X):
        if self.use_cl:
            L_feature_list, cl_loss_L = self.freq_feature(rcs_L_time, rcs_L_fft, hrrp_L, text_L)
            S_feature_list, cl_loss_S = self.freq_feature(rcs_S_time, rcs_S_fft, hrrp_S, text_S)
            X_feature_list, cl_loss_X = self.freq_feature(rcs_X_time, rcs_X_fft, hrrp_X, text_X)
            cl_loss = 0.3*cl_loss_L + 0.3*cl_loss_S + 0.4*cl_loss_X
        else:
            L_feature_list, S_feature_list, X_feature_list = self.freq_feature(rcs_L_time, rcs_L_fft, hrrp_L, text_L), self.freq_feature(rcs_S_time, rcs_S_fft, hrrp_S, text_S), self.freq_feature(rcs_X_time, rcs_X_fft, hrrp_X, text_X)
        [rcs_feat_L, hrrp_feat_L, text_feat_L] = L_feature_list # (64, 128)
        [rcs_feat_S, hrrp_feat_S, text_feat_S] = S_feature_list
        [rcs_feat_X, hrrp_feat_X, text_feat_X] = X_feature_list

        fusion_feature = self.gat_fusion(rcs_feat_L, hrrp_feat_L, text_feat_L,
                                         rcs_feat_S, hrrp_feat_S, text_feat_S,
                                         rcs_feat_X, hrrp_feat_X, text_feat_X)
        output = self.fc(fusion_feature)

        if self.use_cl:
            return output, cl_loss
        else:
            return output 


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    rcs_L_time = torch.zeros((64, 1, 101)).to(device=device)
    rcs_L_fft = torch.zeros((64, 1, 101)).to(device=device)
    rcs_S_time = torch.zeros((64, 1, 101)).to(device=device)
    rcs_S_fft = torch.zeros((64, 1, 101)).to(device=device)
    rcs_X_time = torch.zeros((64, 1, 101)).to(device=device)
    rcs_X_fft = torch.zeros((64, 1, 101)).to(device=device)
    # model = Multi_RCS_single_ablation(gat_input_dim=128, gat_hidden_dim=128, gat_output_dim=128)
    model = Multi_RCS_single_ablation_baseline()
    model.to(device)
    rcs_fusion = model(rcs_L_time, rcs_L_fft, rcs_S_time, rcs_S_fft, rcs_X_time, rcs_X_fft) # (64, 128)
    print(rcs_fusion.shape)
        




