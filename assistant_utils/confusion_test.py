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
from thop import profile
from thop import clever_format
from args_file import get_args

from utils.data_utils import Dataset, Dataset_Text
from utils.func_utils import compute_flops_params
from model.models import GATFusion3_Net

args = get_args().parse_args()

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

# 读取3个频段的数据
    freq_list = ['L', 'S', 'X']
    # text_only = False # 文本信息是否只有1种描述方式
    snr = 15 # 0dB数据需要多跑几次才能出现85%以上的准确率
    batch_size = args.batch_size
    use_cl = args.use_cl
    # rcs_L_time
    dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
    train_rcs_L_time = dataset.create_dataloader(type='train')
    test_rcs_L_time = dataset.create_dataloader(type='test')
    # rcs_L_fft
    dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
    train_rcs_L_fft = dataset.create_dataloader(type='train')
    test_rcs_L_fft = dataset.create_dataloader(type='test')
    # hrrp_L
    dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
    train_hrrp_L = dataset.create_dataloader(type='train')
    test_hrrp_L = dataset.create_dataloader(type='test')
    # text_L
    dataset = Dataset_Text(data_path=f'text_creator_v2/data/data_{freq_list[0]}_{snr}dB.npy', batch_size=batch_size)
    train_text_L = dataset.create_dataloader(type='train')
    test_text_L = dataset.create_dataloader(type='test')
    # rcs_S_time
    dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
    train_rcs_S_time = dataset.create_dataloader(type='train')
    test_rcs_S_time = dataset.create_dataloader(type='test')
    # rcs_S_fft
    dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
    train_rcs_S_fft = dataset.create_dataloader(type='train')
    test_rcs_S_fft = dataset.create_dataloader(type='test')
    # hrrp_S
    dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
    train_hrrp_S = dataset.create_dataloader(type='train')
    test_hrrp_S = dataset.create_dataloader(type='test')
    # text_S
    dataset = Dataset_Text(data_path=f'text_creator_v2/data/data_{freq_list[1]}_{snr}dB.npy', batch_size=batch_size)
    train_text_S = dataset.create_dataloader(type='train')
    test_text_S = dataset.create_dataloader(type='test')
    # rcs_X_time
    dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
    train_rcs_X_time = dataset.create_dataloader(type='train')
    test_rcs_X_time = dataset.create_dataloader(type='test')
    # rcs_X_fft
    dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
    train_rcs_X_fft = dataset.create_dataloader(type='train')
    test_rcs_X_fft = dataset.create_dataloader(type='test')
    # hrrp_X
    dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
    train_hrrp_X = dataset.create_dataloader(type='train')
    test_hrrp_X = dataset.create_dataloader(type='test')
    # text_X
    dataset = Dataset_Text(data_path=f'text_creator_v2/data/data_{freq_list[2]}_{snr}dB.npy', batch_size=batch_size)
    train_text_X = dataset.create_dataloader(type='train')
    test_text_X = dataset.create_dataloader(type='test')

    dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
    test_label = dataset.get_test_label()

    model = GATFusion3_Net(rht_output_dim=args.rht_output_dim, gat_hidden_dim=args.gat_hidden_dim, 
                           gat_output_dim=args.gat_output_dim, use_cl=args.use_cl, num_class=args.num_class)
    model_weight = f'weights/proposed_method/15dB/model_acc93.80165289256199.pth'
    model.load_state_dict(torch.load(model_weight))
    model.to(device)
    model.eval()

    test_pred = []
    for batch_idx1, (data_L_rcs_time, data_L_rcs_fft, data_L_hrrp, data_L_text, data_S_rcs_time, data_S_rcs_fft, data_S_hrrp, data_S_text, data_X_rcs_time, data_X_rcs_fft, data_X_hrrp, data_X_text) in tqdm(enumerate(zip(test_rcs_L_time, test_rcs_L_fft, test_hrrp_L, test_text_L, test_rcs_S_time, test_rcs_S_fft, test_hrrp_S, test_text_S, test_rcs_X_time, test_rcs_X_fft, test_hrrp_X, test_text_X)),total=len(test_hrrp_L)):
        inputs_L_rcs_time, labels = data_L_rcs_time
        inputs_L_rcs_fft, _ = data_L_rcs_fft
        inputs_L_hrrp, _ = data_L_hrrp
        inputs_L_text, _ = data_L_text
        inputs_S_rcs_time, _ = data_S_rcs_time
        inputs_S_rcs_fft, _ = data_S_rcs_fft
        inputs_S_hrrp, _ = data_S_hrrp
        inputs_S_text, _ = data_S_text
        inputs_X_rcs_time, _ = data_X_rcs_time
        inputs_X_rcs_fft, _ = data_X_rcs_fft
        inputs_X_hrrp, _ = data_X_hrrp
        inputs_X_text, _ = data_X_text
        inputs_L_rcs_time, inputs_L_rcs_fft, inputs_S_rcs_time, inputs_S_rcs_fft, inputs_X_rcs_time, inputs_X_rcs_fft, labels = inputs_L_rcs_time.to(device), inputs_L_rcs_fft.to(device), inputs_S_rcs_time.to(device), inputs_S_rcs_fft.to(device), inputs_X_rcs_time.to(device), inputs_X_rcs_fft.to(device), labels.to(device)
        inputs_L_hrrp, inputs_S_hrrp, inputs_X_hrrp, labels = inputs_L_hrrp.to(device), inputs_S_hrrp.to(device), inputs_X_hrrp.to(device), labels.to(device)
        inputs_L_text, inputs_S_text, inputs_X_text = inputs_L_text.to(device), inputs_S_text.to(device), inputs_X_text.to(device)
        # labels = torch.tensor(labels, dtype=torch.long)  # 需要将label转换成long类型
        labels = labels.clone().detach()
        outputs, cl_loss = model(inputs_L_rcs_time.float(), inputs_L_rcs_fft.float(), inputs_L_hrrp.float(), inputs_L_text.float(), inputs_S_rcs_time.float(), inputs_S_rcs_fft.float(), inputs_S_hrrp.float(), inputs_S_text.float(), inputs_X_rcs_time.float(), inputs_X_rcs_fft.float(), inputs_X_hrrp.float(), inputs_X_text.float())  # 需要加.float()，否则会报错
        outputs = F.softmax(outputs, dim=1)  # 计算准确率时需要添加softmax
        _, predicted = torch.max(outputs.data, 1)
        test_pred = test_pred + list(predicted.cpu().numpy())
    
    test_true = list(test_label)
    # print(test_true)
    # print(test_pred)
    # print(type(test_true))
    # print(type(test_pred))

    test_true = np.array(test_true).squeeze()  # 强制转换为numpy的array形式
    test_pred = np.array(test_pred)

    # print(test_true)
    # print(test_pred)
    print(test_true.shape)
    print(test_pred.shape)

    C = confusion_matrix(test_true, test_pred)  # 混淆矩阵

    print(C)


