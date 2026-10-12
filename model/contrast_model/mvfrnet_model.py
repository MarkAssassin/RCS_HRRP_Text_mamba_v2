import torch
import torch.nn as nn
import torch.nn.functional as F

# MVFRnet: A Novel High-Accuracy Network for ISAR Air-Target Recognition via Multi-View Fusion

# ===================== 1. 论文核心模块：GCT门控通道注意力机制 =====================
class GCTAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(GCTAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        weight = avg_out + max_out
        return x * weight

# ===================== 2. 论文核心模块：SCU跳跃连接单元 =====================
class SCUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SCUBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.gct = GCTAttention(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.gct(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ===================== 一维RCS特征提取器 (维度：batch,1,101) =====================
class RCSExtractor1D(nn.Module):
    def __init__(self):
        super(RCSExtractor1D, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 101, 256)
        )

    def forward(self, x):
        return self.extractor(x)

# ===================== 二维HRRP特征提取器 (维度：batch,1,101,101) =====================
class HRRPExtractor2D(nn.Module):
    def __init__(self):
        super(HRRPExtractor2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.scu1 = SCUBlock(64, 128, stride=2)
        self.scu2 = SCUBlock(128, 256, stride=2)
        self.scu3 = SCUBlock(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.scu1(x)
        x = self.scu2(x)
        x = self.scu3(x)
        x = self.pool(x)
        return self.flatten(x)

# ===================== MVFRnet 主网络（三频段 + 双模态融合） =====================
class MVFRNet(nn.Module):
    def __init__(self, num_classes=6):
        super(MVFRNet, self).__init__()

        # RCS = 1D (batch,1,101)
        self.rcs_L = RCSExtractor1D()
        self.rcs_S = RCSExtractor1D()
        self.rcs_X = RCSExtractor1D()

        # HRRP = 2D (batch,1,101,101)
        self.hrrp_L = HRRPExtractor2D()
        self.hrrp_S = HRRPExtractor2D()
        self.hrrp_X = HRRPExtractor2D()

        # 融合层：3×RCS + 3×HRRP = 3×256 + 3×512 = 2304
        self.fusion = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, rcs_L, hrrp_L, rcs_S, hrrp_S, rcs_X, hrrp_X):
        # 单频段特征
        f_rcs_L = self.rcs_L(rcs_L)
        f_rcs_S = self.rcs_S(rcs_S)
        f_rcs_X = self.rcs_X(rcs_X)
        
        f_hrrp_L = self.hrrp_L(hrrp_L)
        f_hrrp_S = self.hrrp_S(hrrp_S)
        f_hrrp_X = self.hrrp_X(hrrp_X)

        # 多模态 + 多频段融合
        fuse_feat = torch.cat([
            f_rcs_L, f_rcs_S, f_rcs_X,
            f_hrrp_L, f_hrrp_S, f_hrrp_X
        ], dim=1)

        fuse_feat = self.fusion(fuse_feat)
        out = self.classifier(fuse_feat)
        return out

# ===================== 测试代码（维度100%匹配你的数据） =====================
if __name__ == '__main__':
    batch_size = 4

    # RCS：1维 (batch,1,101)
    rcs = torch.randn(batch_size, 1, 101)
    # HRRP：2维 (batch,1,101,101)
    hrrp = torch.randn(batch_size, 1, 101, 101)

    model = MVFRNet(num_classes=6)
    output = model(rcs, hrrp, rcs, hrrp, rcs, hrrp)

    print(f"批次大小: {batch_size}")
    print(f"输出维度: {output.shape}")  # 输出 (4,6)，正确
    print("✅ 网络维度完全正确，构建成功！")