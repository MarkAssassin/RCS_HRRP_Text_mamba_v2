import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import math

# from rh_fusion import RHFusion_Net
# from GRU import GRU
# from GCN import GCN
# from GAT import build_graph_from_modalities_3modal, GATFusion3

from model.rht_fusion import RHTAlign_Net
from model.gat import GATFusion3_v3, GATFusion3_v3_nopAll

class RHT_Net(nn.Module):
    def __init__(self, use_cl=True):
        super(RHT_Net, self).__init__()
        self.use_cl = use_cl
        self.fusion_net = RHTAlign_Net(use_cl=use_cl)
        
    def forward(self, rcs_time, rcs_fft, hrrp, text):
        if self.use_cl:
            feature_list, cl_loss = self.fusion_net(rcs_time, rcs_fft, hrrp, text)
            return feature_list, cl_loss
        else:
            feature_list = self.fusion_net(rcs_time, rcs_fft, hrrp, text)
            return feature_list

class GATFusion3_Net(nn.Module):
    def __init__(self, rht_output_dim, gat_hidden_dim, gat_output_dim, use_cl=True, num_class=4):
        super().__init__()

        # 三个频段的特征抽取器（你可自行换成 GRU/CNN）
        # L/S/X频段rcs&hrrp&text融合特征
        self.use_cl = use_cl
        self.freq_feature = RHT_Net(use_cl=use_cl) # (batch, gru_output_dim)
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