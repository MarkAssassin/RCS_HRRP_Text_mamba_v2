import torch
import torch.nn as nn
import torch.nn.functional as F

# JOINT TARGET RECOGNITION FOR MULTI-STATION ISAR VIA MIIR NETWORK

# ==================== 1. 论文核心模块：GCT 注意力 ================
class GCT(nn.Module):
    """门控通道变换注意力（论文MIIR Network核心模块）"""
    def __init__(self, channels, ratio=16):
        super(GCT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ==================== 2. 论文核心模块：SCU 跳跃连接单元 ================
class SCU(nn.Module):
    """跳跃连接单元（防止梯度消失，强化深层特征）"""
    def __init__(self, in_channels, out_channels):
        super(SCU, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

# ==================== 3. 一维 RCS 特征提取器 ================
class RCSBranch(nn.Module):
    """输入：(B,1,101)"""
    def __init__(self):
        super(RCSBranch, self).__init__()
        self.branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Flatten()
        )
        # 输出维度：32*101 = 3232

    def forward(self, x):
        return self.branch(x)

# ==================== 4. 二维 HRRP 特征提取器（MIIR 结构） ================
class HRRPBranch(nn.Module):
    """输入：(B,1,101,101)"""
    def __init__(self):
        super(HRRPBranch, self).__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.scu1 = SCU(32, 64)
        self.gct1 = GCT(64)
        self.scu2 = SCU(64, 128)
        self.gct2 = GCT(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        # 输出维度：128

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.scu1(x)
        x = self.gct1(x)
        x = self.scu2(x)
        x = self.gct2(x)
        x = self.pool(x)
        return self.flatten(x)

# ==================== 5. 多频段 + 双模态 融合网络（最终模型） ================
class MIIR_MultiBand_Fusion(nn.Module):
    def __init__(self, num_classes=6):
        super(MIIR_MultiBand_Fusion, self).__init__()
        # 3个频段独立分支 L / S / X
        self.rcs_L = RCSBranch()
        self.rcs_S = RCSBranch()
        self.rcs_X = RCSBranch()

        self.hrrp_L = HRRPBranch()
        self.hrrp_S = HRRPBranch()
        self.hrrp_X = HRRPBranch()

        # ==================== 修复维度！====================
        # 总维度计算：
        # RCS: 3 × 3232 = 9696
        # HRRP: 3 × 128 = 384
        # 总计：9696+384 = 10080
        self.fc = nn.Sequential(
            nn.Linear(10080, 512),  # 这里修复为 10080
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, rcs_L, hrrp_L, rcs_S, hrrp_S, rcs_X, hrrp_X):
        # RCS 特征
        f_rcs_L = self.rcs_L(rcs_L)
        f_rcs_S = self.rcs_S(rcs_S)
        f_rcs_X = self.rcs_X(rcs_X)

        # HRRP 特征
        f_hrrp_L = self.hrrp_L(hrrp_L)
        f_hrrp_S = self.hrrp_S(hrrp_S)
        f_hrrp_X = self.hrrp_X(hrrp_X)

        # 拼接所有特征
        feat = torch.cat([
            f_rcs_L, f_hrrp_L,
            f_rcs_S, f_hrrp_S,
            f_rcs_X, f_hrrp_X
        ], dim=1)

        out = self.fc(feat)
        return out

# ==================== 测试网络是否匹配你的数据维度 ================
if __name__ == "__main__":
    batch = 4
    rcs = torch.randn(batch, 1, 101)       # RCS: (B,1,101)
    hrrp = torch.randn(batch, 1, 101, 101) # HRRP: (B,1,101,101)

    model = MIIR_MultiBand_Fusion(num_classes=6)
    output = model(rcs, hrrp, rcs, hrrp, rcs, hrrp)

    print("输出形状:", output.shape)  # 正确输出 torch.Size([4,6])
    print("网络构建成功 ✅ 无报错！")