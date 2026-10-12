'''
    使用时将其移至最外层运行
'''
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
import torchextractor as tx # PyTorch Intermediate Feature Extraction
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils.data_utils import Dataset, Dataset_Text
from utils.draw_utils import draw_SNE
from model.models import GATFusion3_Net
from model.ablation_model import Multi_RCS_single_ablation, Multi_HRRP_single_ablation, Multi_RCS_Text_double_ablation, Multi_RCS_HRRP_double_ablation, Multi_HRRP_Text_double_ablation
from model.contrast_model.mdfl_model import GATFusion3_Net_Cat, GATFusion3_Net_Plus, GATFusion3_Net_Trans, RadarMultiDomainNet
from model.contrast_model.mvfrnet_model import MVFRNet
from args_file import get_args

args = get_args().parse_args()

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
##################################### 多频段RCS融合特征 #####################################
# 读取3个频段的数据
    # freq_list = ['L', 'S', 'X']
    # snr = 20 # 0dB数据需要多跑几次才能出现85%以上的准确率
    # batch_size = args.batch_size
    # # rcs_L_time
    # dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
    # train_rcs_L_time = dataset.create_dataloader(type='train')
    # test_rcs_L_time = dataset.create_dataloader(type='test')
    # # rcs_L_fft
    # dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
    # train_rcs_L_fft = dataset.create_dataloader(type='train')
    # test_rcs_L_fft = dataset.create_dataloader(type='test')

    # # rcs_S_time
    # dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
    # train_rcs_S_time = dataset.create_dataloader(type='train')
    # test_rcs_S_time = dataset.create_dataloader(type='test')
    # # rcs_S_fft
    # dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
    # train_rcs_S_fft = dataset.create_dataloader(type='train')
    # test_rcs_S_fft = dataset.create_dataloader(type='test')

    # # rcs_X_time
    # dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
    # train_rcs_X_time = dataset.create_dataloader(type='train')
    # test_rcs_X_time = dataset.create_dataloader(type='test')
    # # rcs_X_fft
    # dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
    # train_rcs_X_fft = dataset.create_dataloader(type='train')
    # test_rcs_X_fft = dataset.create_dataloader(type='test')

    # dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
    # test_label = dataset.get_test_label()

    # model = Multi_RCS_single_ablation(gat_input_dim=128, gat_hidden_dim=128, gat_output_dim=128)
    # model_weight = f'weights/ablation_method/multi_rcs/20dB/model_acc92.42424242424242.pth'
    # model.load_state_dict(torch.load(model_weight))
    # model.to(device)
    ## print(tx.list_module_names(model))
    # model = tx.Extractor(model, "fc.1")
    # model.eval()
    # feature_list = []
    # for batch_idx1, (data_L_rcs_time, data_L_rcs_fft, data_S_rcs_time, data_S_rcs_fft, data_X_rcs_time, data_X_rcs_fft) in tqdm(enumerate(zip(test_rcs_L_time, test_rcs_L_fft, test_rcs_S_time, test_rcs_S_fft, test_rcs_X_time, test_rcs_X_fft)),total=len(test_rcs_L_time)):
    #     inputs_L_rcs_time, labels = data_L_rcs_time
    #     inputs_L_rcs_fft, _ = data_L_rcs_fft
    #     inputs_S_rcs_time, _ = data_S_rcs_time
    #     inputs_S_rcs_fft, _ = data_S_rcs_fft
    #     inputs_X_rcs_time, _ = data_X_rcs_time
    #     inputs_X_rcs_fft, _ = data_X_rcs_fft
    #     inputs_L_rcs_time, inputs_L_rcs_fft, inputs_S_rcs_time, inputs_S_rcs_fft, inputs_X_rcs_time, inputs_X_rcs_fft, labels = inputs_L_rcs_time.to(device), inputs_L_rcs_fft.to(device), inputs_S_rcs_time.to(device), inputs_S_rcs_fft.to(device), inputs_X_rcs_time.to(device), inputs_X_rcs_fft.to(device), labels.to(device)
    #     # labels = torch.tensor(labels, dtype=torch.long)  # 需要将label转换成long类型
    #     labels = labels.clone().detach()
    #     _, feature = model(inputs_L_rcs_time.float(), inputs_L_rcs_fft.float(), inputs_S_rcs_time.float(), inputs_S_rcs_fft.float(), inputs_X_rcs_time.float(), inputs_X_rcs_fft.float())  # 需要加.float()，否则会报错
    #     feature = feature['fc.1'] # (batch, 64)
    #     # feature = torch.squeeze(feature)
    #     feature_list.append(feature.cpu().detach().numpy()) 
    # feature_list = np.concatenate(feature_list, axis=0)
    # test_label = test_label
    # test_label = np.array(test_label).squeeze()
    # save_path = 'pictures/t-SNE_multi_rcs.png'
    # draw_SNE(feature_list, test_label, save_path)

##################################### 多频段HRRP融合特征 #####################################
# # 读取3个频段的数据
#     freq_list = ['L', 'S', 'X']
#     snr = 20 # 0dB数据需要多跑几次才能出现85%以上的准确率
#     batch_size = args.batch_size
#     # hrrp_L
#     dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
#     train_hrrp_L = dataset.create_dataloader(type='train')
#     test_hrrp_L = dataset.create_dataloader(type='test')
#     # hrrp_S
#     dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
#     train_hrrp_S = dataset.create_dataloader(type='train')
#     test_hrrp_S = dataset.create_dataloader(type='test')
#     # hrrp_X
#     dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
#     train_hrrp_X = dataset.create_dataloader(type='train')
#     test_hrrp_X = dataset.create_dataloader(type='test')


#     dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
#     test_label = dataset.get_test_label()

#     model = Multi_HRRP_single_ablation(gat_input_dim=128, gat_hidden_dim=128, gat_output_dim=128)
#     model_weight = f'weights/ablation_method/multi_hrrp/20dB/model_acc81.5426997245179.pth'
#     model.load_state_dict(torch.load(model_weight))
#     model.to(device)
#     # print(tx.list_module_names(model))
#     model = tx.Extractor(model, "fc.1")
#     model.eval()
#     feature_list = []
#     for batch_idx, (data_L_hrrp, data_S_hrrp, data_X_hrrp) in tqdm(enumerate(zip(test_hrrp_L, test_hrrp_S, test_hrrp_X)),total=len(test_hrrp_L)):
#         inputs_L_hrrp, labels = data_L_hrrp
#         inputs_S_hrrp, _ = data_S_hrrp
#         inputs_X_hrrp, _ = data_X_hrrp
#         inputs_L_hrrp, inputs_S_hrrp, inputs_X_hrrp, labels = inputs_L_hrrp.to(device), inputs_S_hrrp.to(device), inputs_X_hrrp.to(device), labels.to(device)
#         # labels = torch.tensor(labels, dtype=torch.long)  # 需要将label转换成long类型
#         labels = labels.clone().detach()
#         _, feature = model(inputs_L_hrrp.float(), inputs_S_hrrp.float(), inputs_X_hrrp.float())  # 需要加.float()，否则会报错
#         feature = feature['fc.1'] # (batch, 64)
#         # feature = torch.squeeze(feature)
#         feature_list.append(feature.cpu().detach().numpy()) 
#     feature_list = np.concatenate(feature_list, axis=0)
#     test_label = test_label
#     test_label = np.array(test_label).squeeze()
#     save_path = 'pictures/t-SNE_multi_hrrp.png'
#     draw_SNE(feature_list, test_label, save_path)

##################################### 多频段RCS+Text融合特征 #####################################
# # 读取3个频段的数据
#     freq_list = ['L', 'S', 'X']
#     snr = 20 # 0dB数据需要多跑几次才能出现85%以上的准确率
#     batch_size = args.batch_size
#     # rcs_L_time
#     dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
#     train_rcs_L_time = dataset.create_dataloader(type='train')
#     test_rcs_L_time = dataset.create_dataloader(type='test')
#     # rcs_L_fft
#     dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
#     train_rcs_L_fft = dataset.create_dataloader(type='train')
#     test_rcs_L_fft = dataset.create_dataloader(type='test')
#     # text_L
#     dataset = Dataset_Text(data_path=f'text_creator_v2/data/data_{freq_list[0]}_{snr}dB.npy', batch_size=batch_size)
#     train_text_L = dataset.create_dataloader(type='train')
#     test_text_L = dataset.create_dataloader(type='test')
#     # rcs_S_time
#     dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
#     train_rcs_S_time = dataset.create_dataloader(type='train')
#     test_rcs_S_time = dataset.create_dataloader(type='test')
#     # rcs_S_fft
#     dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
#     train_rcs_S_fft = dataset.create_dataloader(type='train')
#     test_rcs_S_fft = dataset.create_dataloader(type='test')
#     # text_S
#     dataset = Dataset_Text(data_path=f'text_creator_v2/data/data_{freq_list[1]}_{snr}dB.npy', batch_size=batch_size)
#     train_text_S = dataset.create_dataloader(type='train')
#     test_text_S = dataset.create_dataloader(type='test')
#     # rcs_X_time
#     dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
#     train_rcs_X_time = dataset.create_dataloader(type='train')
#     test_rcs_X_time = dataset.create_dataloader(type='test')
#     # rcs_X_fft
#     dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
#     train_rcs_X_fft = dataset.create_dataloader(type='train')
#     test_rcs_X_fft = dataset.create_dataloader(type='test')
#     # text_X
#     dataset = Dataset_Text(data_path=f'text_creator_v2/data/data_{freq_list[2]}_{snr}dB.npy', batch_size=batch_size)
#     train_text_X = dataset.create_dataloader(type='train')
#     test_text_X = dataset.create_dataloader(type='test')


#     dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
#     test_label = dataset.get_test_label()

#     model = Multi_RCS_Text_double_ablation(gat_input_dim=128, gat_hidden_dim=128, gat_output_dim=128, use_cl=True)
#     model_weight = f'weights/ablation_method/multi_rcs_text/20dB/model_acc92.56198347107438.pth'
#     model.load_state_dict(torch.load(model_weight))
#     model.to(device)
#     # print(tx.list_module_names(model))
#     model = tx.Extractor(model, "fc.2")
#     model.eval()
#     feature_list = []
#     for batch_idx, (data_L_rcs_time, data_L_rcs_fft, data_L_text, data_S_rcs_time, data_S_rcs_fft, data_S_text, data_X_rcs_time, data_X_rcs_fft, data_X_text) in tqdm(enumerate(zip(test_rcs_L_time, test_rcs_L_fft, test_text_L, test_rcs_S_time, test_rcs_S_fft, test_text_S, test_rcs_X_time, test_rcs_X_fft, test_text_X)),total=len(test_rcs_L_time)):
#         inputs_L_rcs_time, labels = data_L_rcs_time
#         inputs_L_rcs_fft, _ = data_L_rcs_fft
#         inputs_L_text, _ = data_L_text
#         inputs_S_rcs_time, _ = data_S_rcs_time
#         inputs_S_rcs_fft, _ = data_S_rcs_fft
#         inputs_S_text, _ = data_S_text
#         inputs_X_rcs_time, _ = data_X_rcs_time
#         inputs_X_rcs_fft, _ = data_X_rcs_fft
#         inputs_X_text, _ = data_X_text
#         inputs_L_rcs_time, inputs_L_rcs_fft, inputs_S_rcs_time, inputs_S_rcs_fft, inputs_X_rcs_time, inputs_X_rcs_fft, labels = inputs_L_rcs_time.to(device), inputs_L_rcs_fft.to(device), inputs_S_rcs_time.to(device), inputs_S_rcs_fft.to(device), inputs_X_rcs_time.to(device), inputs_X_rcs_fft.to(device), labels.to(device)
#         inputs_L_text, inputs_S_text, inputs_X_text = inputs_L_text.to(device), inputs_S_text.to(device), inputs_X_text.to(device)
#         # labels = torch.tensor(labels, dtype=torch.long)  # 需要将label转换成long类型
#         labels = labels.clone().detach()
#         _, feature = model(inputs_L_rcs_time.float(), inputs_L_rcs_fft.float(), inputs_L_text.float(), inputs_S_rcs_time.float(), inputs_S_rcs_fft.float(), inputs_S_text.float(), inputs_X_rcs_time.float(), inputs_X_rcs_fft.float(), inputs_X_text.float())  # 需要加.float()，否则会报错
#         feature = feature['fc.2'] # (batch, 64)
#         # feature = torch.squeeze(feature)
#         feature_list.append(feature.cpu().detach().numpy()) 
#     feature_list = np.concatenate(feature_list, axis=0)
#     test_label = test_label
#     test_label = np.array(test_label).squeeze()
#     # save_path = 'pictures/t-SNE_multi_rcs_text.png'
#     save_path = 'pictures/t-SNE_LSX.png'
#     draw_SNE(feature_list, test_label, save_path)

# ##################################### 多频段RCS+HRRP融合特征 #####################################
# # 读取3个频段的数据
#     freq_list = ['L', 'S', 'X']
#     snr = 20 # 0dB数据需要多跑几次才能出现85%以上的准确率
#     batch_size = args.batch_size
#     # rcs_L_time
#     dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
#     train_rcs_L_time = dataset.create_dataloader(type='train')
#     test_rcs_L_time = dataset.create_dataloader(type='test')
#     # rcs_L_fft
#     dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
#     train_rcs_L_fft = dataset.create_dataloader(type='train')
#     test_rcs_L_fft = dataset.create_dataloader(type='test')
#     # hrrp_L
#     dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
#     train_hrrp_L = dataset.create_dataloader(type='train')
#     test_hrrp_L = dataset.create_dataloader(type='test')
#     # rcs_S_time
#     dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
#     train_rcs_S_time = dataset.create_dataloader(type='train')
#     test_rcs_S_time = dataset.create_dataloader(type='test')
#     # rcs_S_fft
#     dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
#     train_rcs_S_fft = dataset.create_dataloader(type='train')
#     test_rcs_S_fft = dataset.create_dataloader(type='test')
#     # hrrp_S
#     dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
#     train_hrrp_S = dataset.create_dataloader(type='train')
#     test_hrrp_S = dataset.create_dataloader(type='test')
#     # rcs_X_time
#     dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
#     train_rcs_X_time = dataset.create_dataloader(type='train')
#     test_rcs_X_time = dataset.create_dataloader(type='test')
#     # rcs_X_fft
#     dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
#     train_rcs_X_fft = dataset.create_dataloader(type='train')
#     test_rcs_X_fft = dataset.create_dataloader(type='test')
#     # hrrp_X
#     dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
#     train_hrrp_X = dataset.create_dataloader(type='train')
#     test_hrrp_X = dataset.create_dataloader(type='test')

#     dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
#     test_label = dataset.get_test_label()

#     model = Multi_RCS_HRRP_double_ablation(gat_input_dim=128, gat_hidden_dim=128, gat_output_dim=128, use_cl=True)
#     model_weight = f'weights/ablation_method/multi_rcs_hrrp/20dB/model_acc80.37190082644628.pth'
#     model.load_state_dict(torch.load(model_weight))
#     model.to(device)
#     # print(tx.list_module_names(model))
#     model = tx.Extractor(model, "fc.1")
#     model.eval()
#     feature_list = []
#     for batch_idx1, (data_L_rcs_time, data_L_rcs_fft, data_L_hrrp, data_S_rcs_time, data_S_rcs_fft, data_S_hrrp, data_X_rcs_time, data_X_rcs_fft, data_X_hrrp) in tqdm(enumerate(zip(test_rcs_L_time, test_rcs_L_fft, test_hrrp_L, test_rcs_S_time, test_rcs_S_fft, test_hrrp_S, test_rcs_X_time, test_rcs_X_fft, test_hrrp_X)),total=len(test_rcs_L_time)):
#         inputs_L_rcs_time, labels = data_L_rcs_time
#         inputs_L_rcs_fft, _ = data_L_rcs_fft
#         inputs_L_hrrp, _ = data_L_hrrp
#         inputs_S_rcs_time, _ = data_S_rcs_time
#         inputs_S_rcs_fft, _ = data_S_rcs_fft
#         inputs_S_hrrp, _ = data_S_hrrp
#         inputs_X_rcs_time, _ = data_X_rcs_time
#         inputs_X_rcs_fft, _ = data_X_rcs_fft
#         inputs_X_hrrp, _ = data_X_hrrp
#         inputs_L_rcs_time, inputs_L_rcs_fft, inputs_S_rcs_time, inputs_S_rcs_fft, inputs_X_rcs_time, inputs_X_rcs_fft, labels = inputs_L_rcs_time.to(device), inputs_L_rcs_fft.to(device), inputs_S_rcs_time.to(device), inputs_S_rcs_fft.to(device), inputs_X_rcs_time.to(device), inputs_X_rcs_fft.to(device), labels.to(device)
#         inputs_L_hrrp, inputs_S_hrrp, inputs_X_hrrp = inputs_L_hrrp.to(device), inputs_S_hrrp.to(device), inputs_X_hrrp.to(device)
#         # labels = torch.tensor(labels, dtype=torch.long)  # 需要将label转换成long类型
#         labels = labels.clone().detach()
#         _, feature = model(inputs_L_rcs_time.float(), inputs_L_rcs_fft.float(), inputs_L_hrrp.float(), inputs_S_rcs_time.float(), inputs_S_rcs_fft.float(), inputs_S_hrrp.float(), inputs_X_rcs_time.float(), inputs_X_rcs_fft.float(), inputs_X_hrrp.float())  # 需要加.float()，否则会报错
#         feature = feature['fc.1'] # (batch, 64)
#         # feature = torch.squeeze(feature)
#         feature_list.append(feature.cpu().detach().numpy()) 
#     feature_list = np.concatenate(feature_list, axis=0)
#     test_label = test_label
#     test_label = np.array(test_label).squeeze()
#     save_path = 'pictures/t-SNE_multi_rcs_hrrp.png'
#     draw_SNE(feature_list, test_label, save_path)

##################################### 多频段HRRP_Text融合特征 #####################################
# # 读取3个频段的数据
#     freq_list = ['L', 'S', 'X']
#     snr = 20 # 0dB数据需要多跑几次才能出现85%以上的准确率
#     batch_size = args.batch_size
#     # hrrp_L
#     dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
#     train_hrrp_L = dataset.create_dataloader(type='train')
#     test_hrrp_L = dataset.create_dataloader(type='test')
#     # text_L
#     dataset = Dataset_Text(data_path=f'text_creator_v2/data/data_{freq_list[0]}_{snr}dB.npy', batch_size=batch_size)
#     train_text_L = dataset.create_dataloader(type='train')
#     test_text_L = dataset.create_dataloader(type='test')
#     # hrrp_S
#     dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
#     train_hrrp_S = dataset.create_dataloader(type='train')
#     test_hrrp_S = dataset.create_dataloader(type='test')
#     # text_S
#     dataset = Dataset_Text(data_path=f'text_creator_v2/data/data_{freq_list[1]}_{snr}dB.npy', batch_size=batch_size)
#     train_text_S = dataset.create_dataloader(type='train')
#     test_text_S = dataset.create_dataloader(type='test')
#     # hrrp_X
#     dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
#     train_hrrp_X = dataset.create_dataloader(type='train')
#     test_hrrp_X = dataset.create_dataloader(type='test')
#     # text_X
#     dataset = Dataset_Text(data_path=f'text_creator_v2/data/data_{freq_list[2]}_{snr}dB.npy', batch_size=batch_size)
#     train_text_X = dataset.create_dataloader(type='train')
#     test_text_X = dataset.create_dataloader(type='test')

#     dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
#     test_label = dataset.get_test_label()

#     model = Multi_HRRP_Text_double_ablation(gat_input_dim=128, gat_hidden_dim=128, gat_output_dim=128, use_cl=True)
#     model_weight = f'weights/ablation_method/multi_hrrp_text/20dB/model_acc75.55096418732782.pth'
#     model.load_state_dict(torch.load(model_weight))
#     model.to(device)
#     # print(tx.list_module_names(model))
#     model = tx.Extractor(model, "fc.1")
#     model.eval()
#     feature_list = []
#     for batch_idx1, (data_L_hrrp, data_L_text, data_S_hrrp, data_S_text, data_X_hrrp, data_X_text) in tqdm(enumerate(zip(test_hrrp_L, test_text_L, test_hrrp_S, test_text_S, test_hrrp_X, test_text_X)),total=len(test_hrrp_L)):
#         inputs_L_hrrp, labels = data_L_hrrp
#         inputs_L_text, _ = data_L_text
#         inputs_S_hrrp, _ = data_S_hrrp
#         inputs_S_text, _ = data_S_text
#         inputs_X_hrrp, _ = data_X_hrrp
#         inputs_X_text, _ = data_X_text
#         inputs_X_text, _ = data_X_text
#         inputs_L_hrrp, inputs_S_hrrp, inputs_X_hrrp, labels = inputs_L_hrrp.to(device), inputs_S_hrrp.to(device), inputs_X_hrrp.to(device), labels.to(device)
#         inputs_L_text, inputs_S_text, inputs_X_text = inputs_L_text.to(device), inputs_S_text.to(device), inputs_X_text.to(device)
#         # labels = torch.tensor(labels, dtype=torch.long)  # 需要将label转换成long类型
#         labels = labels.clone().detach()
#         # labels = torch.tensor(labels, dtype=torch.long)  # 需要将label转换成long类型
#         labels = labels.clone().detach()
#         _, feature = model(inputs_L_hrrp.float(), inputs_L_text.float(), inputs_S_hrrp.float(), inputs_S_text.float(), inputs_X_hrrp.float(), inputs_X_text.float())  # 需要加.float()，否则会报错
#         feature = feature['fc.1'] # (batch, 64)
#         # feature = torch.squeeze(feature)
#         feature_list.append(feature.cpu().detach().numpy()) 
#     feature_list = np.concatenate(feature_list, axis=0)
#     test_label = test_label
#     test_label = np.array(test_label).squeeze()
#     save_path = 'pictures/t-SNE_multi_hrrp_text.png'
#     draw_SNE(feature_list, test_label, save_path)

##################################### 多频段融合特征 #####################################
    # 读取3个频段的数据
    # freq_list = ['L', 'S', 'X']
    # snr = 20 # 0dB数据需要多跑几次才能出现85%以上的准确率
    # batch_size = args.batch_size
    # use_cl = args.use_cl
    # # rcs_L_time
    # dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
    # train_rcs_L_time = dataset.create_dataloader(type='train')
    # test_rcs_L_time = dataset.create_dataloader(type='test')
    # # rcs_L_fft
    # dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
    # train_rcs_L_fft = dataset.create_dataloader(type='train')
    # test_rcs_L_fft = dataset.create_dataloader(type='test')
    # # hrrp_L
    # dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
    # train_hrrp_L = dataset.create_dataloader(type='train')
    # test_hrrp_L = dataset.create_dataloader(type='test')
    # # text_L
    # dataset = Dataset_Text(data_path=f'text_creator_v2/data/data_{freq_list[0]}_{snr}dB.npy', batch_size=batch_size)
    # train_text_L = dataset.create_dataloader(type='train')
    # test_text_L = dataset.create_dataloader(type='test')
    # # rcs_S_time
    # dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
    # train_rcs_S_time = dataset.create_dataloader(type='train')
    # test_rcs_S_time = dataset.create_dataloader(type='test')
    # # rcs_S_fft
    # dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
    # train_rcs_S_fft = dataset.create_dataloader(type='train')
    # test_rcs_S_fft = dataset.create_dataloader(type='test')
    # # hrrp_S
    # dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
    # train_hrrp_S = dataset.create_dataloader(type='train')
    # test_hrrp_S = dataset.create_dataloader(type='test')
    # # text_S
    # dataset = Dataset_Text(data_path=f'text_creator_v2/data/data_{freq_list[1]}_{snr}dB.npy', batch_size=batch_size)
    # train_text_S = dataset.create_dataloader(type='train')
    # test_text_S = dataset.create_dataloader(type='test')
    # # rcs_X_time
    # dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
    # train_rcs_X_time = dataset.create_dataloader(type='train')
    # test_rcs_X_time = dataset.create_dataloader(type='test')
    # # rcs_X_fft
    # dataset = Dataset(data_path=f'dataset/RCS_data_fft_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
    # train_rcs_X_fft = dataset.create_dataloader(type='train')
    # test_rcs_X_fft = dataset.create_dataloader(type='test')
    # # hrrp_X
    # dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
    # train_hrrp_X = dataset.create_dataloader(type='train')
    # test_hrrp_X = dataset.create_dataloader(type='test')
    # # text_X
    # dataset = Dataset_Text(data_path=f'text_creator_v2/data/data_{freq_list[2]}_{snr}dB.npy', batch_size=batch_size)
    # train_text_X = dataset.create_dataloader(type='train')
    # test_text_X = dataset.create_dataloader(type='test')

    # dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
    # test_label = dataset.get_test_label()
    # train_label = dataset.get_train_label

    # # model = GATFusion3_Net(rht_output_dim=args.rht_output_dim, gat_hidden_dim=args.gat_hidden_dim, 
    # #                        gat_output_dim=args.gat_output_dim, use_cl=args.use_cl, num_class=args.num_class)
    # # model = GATFusion3_Net_Cat(use_cl=args.use_cl, num_class=args.num_class)
    # # model = GATFusion3_Net_Plus(use_cl=args.use_cl, num_class=args.num_class)
    # model = GATFusion3_Net_Trans(use_cl=args.use_cl, num_class=args.num_class)

    # # model_weight = f'weights/proposed_method/20dB/model_acc95.45454545454545.pth'
    # # model_weight = f'weights/contrast_method/cat/20dB/model_acc88.84297520661157.pth'
    # # model_weight = f'weights/contrast_method/plus/20dB/model_acc78.92561983471074.pth'
    # model_weight = f'weights/contrast_method/transformer/20dB/model_acc85.74380165289256.pth'
    # model.load_state_dict(torch.load(model_weight))
    # model.to(device)
    # print(tx.list_module_names(model))
    # model = tx.Extractor(model, "fc.1")
    # model.eval()
    # feature_list = []
    # for batch_idx1, (data_L_rcs_time, data_L_rcs_fft, data_L_hrrp, data_L_text, data_S_rcs_time, data_S_rcs_fft, data_S_hrrp, data_S_text, data_X_rcs_time, data_X_rcs_fft, data_X_hrrp, data_X_text) in tqdm(enumerate(zip(test_rcs_L_time, test_rcs_L_fft, test_hrrp_L, test_text_L, test_rcs_S_time, test_rcs_S_fft, test_hrrp_S, test_text_S, test_rcs_X_time, test_rcs_X_fft, test_hrrp_X, test_text_X)),total=len(test_hrrp_L)):
    # # for batch_idx, (data_L_rcs_time, data_L_rcs_fft, data_L_hrrp, data_L_text, data_S_rcs_time, data_S_rcs_fft, data_S_hrrp, data_S_text, data_X_rcs_time, data_X_rcs_fft, data_X_hrrp, data_X_text) in tqdm(enumerate(zip(train_rcs_L_time, train_rcs_L_fft, train_hrrp_L, train_text_L, train_rcs_S_time, train_rcs_S_fft, train_hrrp_S, train_text_S, train_rcs_X_time, train_rcs_X_fft, train_hrrp_X, train_text_X)),total=len(train_hrrp_L)):
    #     inputs_L_rcs_time, labels = data_L_rcs_time
    #     inputs_L_rcs_fft, _ = data_L_rcs_fft
    #     inputs_L_hrrp, _ = data_L_hrrp
    #     inputs_L_text, _ = data_L_text
    #     inputs_S_rcs_time, _ = data_S_rcs_time
    #     inputs_S_rcs_fft, _ = data_S_rcs_fft
    #     inputs_S_hrrp, _ = data_S_hrrp
    #     inputs_S_text, _ = data_S_text
    #     inputs_X_rcs_time, _ = data_X_rcs_time
    #     inputs_X_rcs_fft, _ = data_X_rcs_fft
    #     inputs_X_hrrp, _ = data_X_hrrp
    #     inputs_X_text, _ = data_X_text
    #     inputs_L_rcs_time, inputs_L_rcs_fft, inputs_S_rcs_time, inputs_S_rcs_fft, inputs_X_rcs_time, inputs_X_rcs_fft, labels = inputs_L_rcs_time.to(device), inputs_L_rcs_fft.to(device), inputs_S_rcs_time.to(device), inputs_S_rcs_fft.to(device), inputs_X_rcs_time.to(device), inputs_X_rcs_fft.to(device), labels.to(device)
    #     inputs_L_hrrp, inputs_S_hrrp, inputs_X_hrrp, labels = inputs_L_hrrp.to(device), inputs_S_hrrp.to(device), inputs_X_hrrp.to(device), labels.to(device)
    #     inputs_L_text, inputs_S_text, inputs_X_text = inputs_L_text.to(device), inputs_S_text.to(device), inputs_X_text.to(device)
    #     # labels = torch.tensor(labels, dtype=torch.long)  # 需要将label转换成long类型
    #     labels = labels.clone().detach()
    #     _, feature = model(inputs_L_rcs_time.float(), inputs_L_rcs_fft.float(), inputs_L_hrrp.float(), inputs_L_text.float(), inputs_S_rcs_time.float(), inputs_S_rcs_fft.float(), inputs_S_hrrp.float(), inputs_S_text.float(), inputs_X_rcs_time.float(), inputs_X_rcs_fft.float(), inputs_X_hrrp.float(), inputs_X_text.float())  # 需要加.float()，否则会报错
    #     feature = feature['fc.1'] # (batch, 64)
    #     # feature = torch.squeeze(feature)
    #     feature_list.append(feature.cpu().detach().numpy()) 
    # feature_list = np.concatenate(feature_list, axis=0)
    # test_label = test_label
    # test_label = np.array(test_label).squeeze()
    # # save_path = 'pictures/t-SNE_LSX.png'
    # # save_path = 'pictures/t-SNE_contrast_cat.png'
    # # save_path = 'pictures/t-SNE_contrast_plus.png'
    # save_path = 'pictures/t-SNE_contrast_transformer.png'
    # draw_SNE(feature_list, test_label, save_path)

##################################### 多频段融合特征(mdfl) #####################################
# 读取3个频段的数据
    freq_list = ['L', 'S', 'X']
    # text_only = False # 文本信息是否只有1种描述方式
    snr = 0 # 0dB数据需要多跑几次才能出现85%以上的准确率
    batch_size = 64
    # rcs_L_time
    dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
    train_rcs_L = dataset.create_dataloader(type='train')
    test_rcs_L = dataset.create_dataloader(type='test')
    # hrrp_L
    dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
    train_hrrp_L = dataset.create_dataloader(type='train')
    test_hrrp_L = dataset.create_dataloader(type='test')
    # rcs_S_time
    dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
    train_rcs_S = dataset.create_dataloader(type='train')
    test_rcs_S = dataset.create_dataloader(type='test')
    # hrrp_S
    dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[1]}_{snr}dB.mat', batch_size=batch_size)
    train_hrrp_S = dataset.create_dataloader(type='train')
    test_hrrp_S = dataset.create_dataloader(type='test')
    # rcs_X_time
    dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
    train_rcs_X = dataset.create_dataloader(type='train')
    test_rcs_X = dataset.create_dataloader(type='test')
    # hrrp_X
    dataset = Dataset(data_path=f'dataset/HRRP_data_{freq_list[2]}_{snr}dB.mat', batch_size=batch_size)
    train_hrrp_X = dataset.create_dataloader(type='train')
    test_hrrp_X = dataset.create_dataloader(type='test')

    dataset = Dataset(data_path=f'dataset/RCS_data_{freq_list[0]}_{snr}dB.mat', batch_size=batch_size)
    test_label = dataset.get_test_label()

    # model = RadarMultiDomainNet(num_classes=4)
    model = MVFRNet(num_classes=4)
    model_weight = f'weights/contrast_method/mvfrnet/20dB/model_acc91.2534435261708.pth'
    model.load_state_dict(torch.load(model_weight))
    model.to(device)
    print(tx.list_module_names(model))
    model = tx.Extractor(model, "fusion.4")
    model.eval()
    feature_list = []
    for batch_idx1, (data_L_rcs, data_L_hrrp, data_S_rcs, data_S_hrrp, data_X_rcs, data_X_hrrp) in tqdm(enumerate(zip(test_rcs_L, test_hrrp_L, test_rcs_S, test_hrrp_S, test_rcs_X, test_hrrp_X)),total=len(test_rcs_L)):
        inputs_L_rcs, labels = data_L_rcs
        inputs_L_hrrp, _ = data_L_hrrp
        inputs_L_hrrp = inputs_L_hrrp.unsqueeze(1) # (64, 101, 101) -> (64, 1, 101, 101)
        inputs_S_rcs, _ = data_S_rcs
        inputs_S_hrrp, _ = data_S_hrrp
        inputs_S_hrrp = inputs_S_hrrp.unsqueeze(1) # (64, 101, 101) -> (64, 1, 101, 101)
        inputs_X_rcs, _ = data_X_rcs
        inputs_X_hrrp, _ = data_X_hrrp
        inputs_X_hrrp = inputs_X_hrrp.unsqueeze(1) # (64, 101, 101) -> (64, 1, 101, 101)
        inputs_L_rcs, inputs_S_rcs, inputs_X_rcs, labels = inputs_L_rcs.to(device), inputs_S_rcs.to(device), inputs_X_rcs.to(device), labels.to(device)
        inputs_L_hrrp, inputs_S_hrrp, inputs_X_hrrp = inputs_L_hrrp.to(device), inputs_S_hrrp.to(device), inputs_X_hrrp.to(device)
        # labels = torch.tensor(labels, dtype=torch.long)  # 需要将label转换成long类型
        labels = labels.clone().detach()
        _, feature = model(inputs_L_rcs.float(), inputs_L_hrrp.float(), inputs_S_rcs.float(), inputs_S_hrrp.float(), inputs_X_rcs.float(), inputs_X_hrrp.float())  # 需要加.float()，否则会报错
        feature = feature['fusion.4'] # (batch, 64)
        # feature = torch.squeeze(feature)
        feature_list.append(feature.cpu().detach().numpy()) 
    feature_list = np.concatenate(feature_list, axis=0)
    test_label = test_label
    test_label = np.array(test_label).squeeze()
    save_path = 'pictures/t-SNE_contrast_mvfrnet.png'
    draw_SNE(feature_list, test_label, save_path)


