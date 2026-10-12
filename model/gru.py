import torch
import torch.nn as nn

# 定义GRU网络
class GRU(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_size = hidden_dim  # 隐层大小
        self.num_layers = num_layers  # gru层数
        self.gru = nn.GRU(feature_dim, hidden_dim, num_layers, batch_first=True) # (B,L,D) (batch_size, seq_length, feature_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.Mish(),
            nn.Linear(int(hidden_dim/2), output_dim)
        )

    def forward(self, x, hidden=None):
        batch_size = x.shape[0] # 获取批次大小
        
        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden
        
        # GRU运算
        output, h_0 = self.gru(x, h_0) 
        
        # # 获取GRU输出的维度信息
        # batch_size, timestep, hidden_size = output.shape
        
        # # 将output变成 batch_size * timestep, hidden_dim
        # output = output.reshape(-1, hidden_size)
        
        # # 全连接层
        # output = self.fc(output)  # 形状为batch_size * timestep, 1
        
        # # 转换维度，用于输出
        # output = output.reshape(timestep, batch_size, -1)[-1]

        # 使用均值池化，而不是最后一帧
        output = output.mean(dim=1)     # (B, H)

        output = self.fc(output)
        
        # 我们只需要返回最后一个时间片的数据即可
        return output

if __name__ == '__main__':
    rcs = torch.zeros((64, 1, 256))
    model = GRU(feature_dim=256, hidden_dim=512, num_layers=2, output_dim=256)
    output = model(rcs)
    print(output.shape)  # torch.Size([64, 256])