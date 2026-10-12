import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import math

from torch_geometric.data import Data

class ParentMLP(nn.Module): # 单频段rcs/hrrp/text得到父节点
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )

    def forward(self, rcs, hrrp, text):
        return self.mlp(torch.cat([rcs, hrrp, text], dim=-1))

def build_hierarchical_graph(
    rcs_L, hrrp_L, text_L,
    rcs_S, hrrp_S, text_S,
    rcs_X, hrrp_X, text_X,
    pL, pS, pX, pAll
):
    """
    所有输入维度: (B, D)
    """
    B, D = rcs_L.shape
    device = rcs_L.device

    # ===== 1️⃣ 节点特征拼接 =====
    # 子节点（9B） + 父节点（4B）
    x = torch.cat([
        rcs_L, hrrp_L, text_L,
        rcs_S, hrrp_S, text_S,
        rcs_X, hrrp_X, text_X,
        pL, pS, pX, pAll
    ], dim=0)   # (13B, D)

    # ===== 2️⃣ 节点索引偏移 =====
    off = {
        'rcs_L': 0,
        'hrrp_L': B,
        'text_L': 2 * B,

        'rcs_S': 3 * B,
        'hrrp_S': 4 * B,
        'text_S': 5 * B,

        'rcs_X': 6 * B,
        'hrrp_X': 7 * B,
        'text_X': 8 * B,

        'pL': 9 * B,
        'pS': 10 * B,
        'pX': 11 * B,
        'pAll': 12 * B
    }

    edge_src, edge_dst = [], []

    for i in range(B):
        # ===== 3️⃣ 子节点 ↔ 单频段父节点 =====
        for band, p in [('L', 'pL'), ('S', 'pS'), ('X', 'pX')]:
            for m in ['rcs', 'hrrp', 'text']:
                u = off[f'{m}_{band}'] + i
                v = off[p] + i
                edge_src += [u, v]
                edge_dst += [v, u]

        # ===== 4️⃣ 单频段父节点 ↔ 多频段父节点 =====
        for p in ['pL', 'pS', 'pX']:
            u = off[p] + i
            v = off['pAll'] + i
            edge_src += [u, v]
            edge_dst += [v, u]

    edge_index = torch.tensor(
        [edge_src, edge_dst],
        dtype=torch.long,
        device=device
    )

    return Data(x=x, edge_index=edge_index)

def build_hierarchical_graph_nopAll(
    rcs_L, hrrp_L, text_L,
    rcs_S, hrrp_S, text_S,
    rcs_X, hrrp_X, text_X,
    pL, pS, pX
):
    """
    所有输入维度: (B, D)
    """
    B, D = rcs_L.shape
    device = rcs_L.device

    # ===== 1️⃣ 节点特征拼接 =====
    # 子节点（9B） + 父节点（3B）
    x = torch.cat([
        rcs_L, hrrp_L, text_L,
        rcs_S, hrrp_S, text_S,
        rcs_X, hrrp_X, text_X,
        pL, pS, pX
    ], dim=0)   # (12B, D)

    # ===== 2️⃣ 节点索引偏移 =====
    off = {
        'rcs_L': 0,
        'hrrp_L': B,
        'text_L': 2 * B,

        'rcs_S': 3 * B,
        'hrrp_S': 4 * B,
        'text_S': 5 * B,

        'rcs_X': 6 * B,
        'hrrp_X': 7 * B,
        'text_X': 8 * B,

        'pL': 9 * B,
        'pS': 10 * B,
        'pX': 11 * B
    }

    edge_src, edge_dst = [], []

    for i in range(B):
        # ===== 子节点 ↔ 单频段父节点 =====
        for band, p in [('L', 'pL'), ('S', 'pS'), ('X', 'pX')]:
            for m in ['rcs', 'hrrp', 'text']:
                u = off[f'{m}_{band}'] + i
                v = off[p] + i
                edge_src += [u, v]
                edge_dst += [v, u]
            
        pL_i = off['pL'] + i
        pS_i = off['pS'] + i
        pX_i = off['pX'] + i

        parent_pairs = [
            (pL_i, pS_i),
            (pS_i, pX_i),
            (pL_i, pX_i)
        ]

        for u, v in parent_pairs:
            edge_src += [u, v]
            edge_dst += [v, u]

    edge_index = torch.tensor(
        [edge_src, edge_dst],
        dtype=torch.long,
        device=device
    )

    return Data(x=x, edge_index=edge_index)

class GAT_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super().__init__()
        # 2层GAT
        self.gat1 = GATConv(
            input_dim, hidden_dim,
            heads=heads, dropout=0.3, concat=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

        self.gat2 = GATConv(
            hidden_dim * heads, output_dim,
            heads=1, dropout=0.3, concat=False
        )
        self.bn2 = nn.BatchNorm1d(output_dim)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        return x

class GATFusion3_v3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super().__init__()

        self.single_parent = ParentMLP(input_dim, hidden_dim)
        self.multi_parent = ParentMLP(input_dim, hidden_dim)

        self.gat = GAT_Net(input_dim, hidden_dim, output_dim, heads)


    def forward(self,
                rcs_L, hrrp_L, text_L,
                rcs_S, hrrp_S, text_S,
                rcs_X, hrrp_X, text_X):

        # ===== 1️⃣ 初始化父节点 =====
        pL = self.single_parent(rcs_L, hrrp_L, text_L)
        pS = self.single_parent(rcs_S, hrrp_S, text_S)
        pX = self.single_parent(rcs_X, hrrp_X, text_X)

        pAll = self.multi_parent(pL, pS, pX)

        # ===== 2️⃣ 构图 =====
        graph = build_hierarchical_graph(
            rcs_L, hrrp_L, text_L,
            rcs_S, hrrp_S, text_S,
            rcs_X, hrrp_X, text_X,
            pL, pS, pX, pAll
        )

        # ===== 3️⃣ GAT =====
        out = self.gat(graph.x, graph.edge_index)

        # ===== 4️⃣ 取多频段父节点 =====
        B = rcs_L.size(0)
        final_parent = out[12 * B: 13 * B]

        return final_parent

class GATFusion3_v3_nopAll(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, feat_type='all'):
        super().__init__()
        self.feat_type = feat_type

        self.single_parent = ParentMLP(input_dim, hidden_dim)

        self.gat = GAT_Net(input_dim, hidden_dim, output_dim, heads)

        self.ln = nn.Linear(output_dim, 1)


    def forward(self,
                rcs_L, hrrp_L, text_L,
                rcs_S, hrrp_S, text_S,
                rcs_X, hrrp_X, text_X):

        # ===== 1️⃣ 初始化父节点 =====
        pL = self.single_parent(rcs_L, hrrp_L, text_L)
        pS = self.single_parent(rcs_S, hrrp_S, text_S)
        pX = self.single_parent(rcs_X, hrrp_X, text_X)

        # ===== 2️⃣ 构图 =====
        graph = build_hierarchical_graph_nopAll(
            rcs_L, hrrp_L, text_L,
            rcs_S, hrrp_S, text_S,
            rcs_X, hrrp_X, text_X,
            pL, pS, pX
        )

        # ===== 3️⃣ GAT =====
        out = self.gat(graph.x, graph.edge_index)

        # ===== 4️⃣ 取多频段父节点 =====
        B = rcs_L.size(0)
        L_feature = out[9*B : 10*B]
        S_feature = out[10*B : 11*B]
        X_feature = out[11*B : 12*B]
        # 原注意力融合不变
        if self.feat_type == 'L':
            return L_feature
        elif self.feat_type == 'S':
            return S_feature
        elif self.feat_type == 'X':
            return X_feature
        elif self.feat_type == 'all':
            # 原注意力融合不变
            L_query = self.ln(L_feature)
            S_query = self.ln(S_feature)
            X_query = self.ln(X_feature)
            # print(f'L:{L_query[0]}/S:{S_query[0]}/X:{X_query[0]}')
            queries = F.softmax(
                torch.cat([L_query, S_query, X_query], dim=1), dim=1
            ).unsqueeze(-1)

            features = torch.stack(
                [L_feature, S_feature, X_feature], dim=1
            )

            fusion_feature = torch.sum(queries * features, dim=1)

            return fusion_feature
        else:
            raise ValueError(f'Unknown feat_type: {self.feat_type}')


########################## 消融实验网络(每个频段单个节点或两个节点) ##########################
# 单个节点
class GATFusion_singleInput(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super().__init__()

        self.gat = GAT_Net(input_dim, hidden_dim, output_dim, heads)
        self.ln = nn.Linear(output_dim, 1)

    def build_graph(self, fL, fS, fX):
        B, D = fL.shape
        device = fL.device

        # 节点: 3B
        x = torch.cat([fL, fS, fX], dim=0)

        off = {
            'L': 0,
            'S': B,
            'X': 2 * B
        }

        edge_src, edge_dst = [], []

        for i in range(B):
            L = off['L'] + i
            S = off['S'] + i
            X = off['X'] + i

            pairs = [(L, S), (S, X), (L, X)]
            for u, v in pairs:
                edge_src += [u, v]
                edge_dst += [v, u]

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
        return Data(x=x, edge_index=edge_index)

    def forward(self, fL, fS, fX):
        graph = self.build_graph(fL, fS, fX)
        out = self.gat(graph.x, graph.edge_index)

        B = fL.size(0)
        L_feat = out[0:B]
        S_feat = out[B:2*B]
        X_feat = out[2*B:3*B]

        # 注意力融合
        query = F.softmax(
            torch.cat([
                self.ln(L_feat),
                self.ln(S_feat),
                self.ln(X_feat)
            ], dim=1),
            dim=1
        ).unsqueeze(-1)

        features = torch.stack([L_feat, S_feat, X_feat], dim=1)
        fusion = torch.sum(query * features, dim=1)

        return fusion


# 两个节点
class ParentMLP_2input(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )

    def forward(self, f1, f2):
        return self.mlp(torch.cat([f1, f2], dim=-1))

class GATFusion_2input(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super().__init__()

        self.parent = ParentMLP_2input(input_dim, hidden_dim)
        self.gat = GAT_Net(input_dim, hidden_dim, output_dim, heads)
        self.ln = nn.Linear(output_dim, 1)

    def forward(self,
                f1_L, f2_L,
                f1_S, f2_S,
                f1_X, f2_X):

        # ===== 父节点 =====
        pL = self.parent(f1_L, f2_L)
        pS = self.parent(f1_S, f2_S)
        pX = self.parent(f1_X, f2_X)

        # ===== 构图（6B子节点 + 3B父节点）=====
        B = f1_L.size(0)
        device = f1_L.device

        x = torch.cat([
            f1_L, f2_L,
            f1_S, f2_S,
            f1_X, f2_X,
            pL, pS, pX
        ], dim=0)

        off = {
            'f1_L': 0, 'f2_L': B,
            'f1_S': 2*B, 'f2_S': 3*B,
            'f1_X': 4*B, 'f2_X': 5*B,
            'pL': 6*B, 'pS': 7*B, 'pX': 8*B
        }

        edge_src, edge_dst = [], []

        for i in range(B):
            for band, p in [('L','pL'),('S','pS'),('X','pX')]:
                for m in ['f1','f2']:
                    u = off[f'{m}_{band}'] + i
                    v = off[p] + i
                    edge_src += [u, v]
                    edge_dst += [v, u]

            # 父节点 fully-connected
            pL_i = off['pL'] + i
            pS_i = off['pS'] + i
            pX_i = off['pX'] + i

            for u,v in [(pL_i,pS_i),(pS_i,pX_i),(pL_i,pX_i)]:
                edge_src += [u,v]
                edge_dst += [v,u]

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)

        # ===== GAT =====
        out = self.gat(x, edge_index)

        L_feat = out[6*B:7*B]
        S_feat = out[7*B:8*B]
        X_feat = out[8*B:9*B]

        # ===== 注意力融合 =====
        query = F.softmax(
            torch.cat([
                self.ln(L_feat),
                self.ln(S_feat),
                self.ln(X_feat)
            ], dim=1),
            dim=1
        ).unsqueeze(-1)

        features = torch.stack([L_feat, S_feat, X_feat], dim=1)
        fusion = torch.sum(query * features, dim=1)

        return fusion



if __name__ == '__main__':
    rcs_L = torch.zeros((64, 128))
    hrrp_L = torch.zeros((64, 128))
    text_L = torch.zeros((64, 128))
    rcs_S = torch.zeros((64, 128))
    hrrp_S = torch.zeros((64, 128))
    text_S = torch.zeros((64, 128))
    rcs_X = torch.zeros((64, 128))
    hrrp_X = torch.zeros((64, 128))
    text_X = torch.zeros((64, 128))
    model = GATFusion3_v3(input_dim=128, hidden_dim=128, output_dim=128)
    final_feat = model(rcs_L, hrrp_L, text_L, rcs_S, hrrp_S, text_S, rcs_X, hrrp_X, text_X)
    print(final_feat.shape)

