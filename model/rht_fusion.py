import torch.nn as nn
import torch.nn.functional as F
import torch

from model.rcs_tf_fusion import Time_Freq_Net_Coupled
from model.cnn import hrrp_cnn
from model.hrrp_mamba import Mamba_Net
# from rcs_tf_fusion import Time_Freq_Net_Coupled
# from cnn import hrrp_cnn
# from hrrp_mamba import Mamba_Net

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

class RHTAlign_Net(nn.Module):
    def __init__(self, d_model=101, split_num=4, temperature=0.5, use_cl=True):
        super(RHTAlign_Net, self).__init__()
        self.use_cl = use_cl
        self.temperature = temperature
        # self.d_feature = int(d_model/split_num)
        self.rcs_fusion = Time_Freq_Net_Coupled()
        # self.hrrp_cnn = hrrp_cnn(feature_dim=d_model)
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
        # hrrp_feature = self.hrrp_cnn(hrrp) # (64, 128)
        # hrrp_feature = hrrp_feature.reshape(-1, 4, 32)
        # hrrp_feature = hrrp_feature + positional_encoding(hrrp_feature, self.d_feature) # (64, 4, 32)
        hrrp_feature = self.hrrp_mamba(hrrp) # (64, 128)
        text_feature = text  # text 已经是 (B, feature_dim) 形式
        
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

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    rcs_time = torch.zeros((64, 1, 101)).to(device)
    rcs_fft = torch.zeros((64, 1, 101)).to(device)
    hrrp = torch.zeros((64, 101, 101)).to(device)
    text = torch.zeros((64, 128))
    model = RHTAlign_Net()
    model.to(device)
    feat_list, cl_loss = model(rcs_time, rcs_fft, hrrp, text)
    print(feat_list[0].shape)  # torch.Size([64, 256])