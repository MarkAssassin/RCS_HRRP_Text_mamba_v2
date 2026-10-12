import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix

from args_file import get_args
from utils.data_utils import Dataset, Dataset_Text
from utils.func_utils import compute_flops_params
from model.contrast_model.mvfrnet_model import MVFRNet

args = get_args().parse_args()

# JOINT TARGET RECOGNITION FOR MULTI-STATION ISAR VIA MIIR NETWORK

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
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

    model = MVFRNet(num_classes=4)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    # # 计算网络的FLOPs和参数量（跑完需要注释掉！）
    # input_rcsdata = torch.randn(64, 1, 101).to(device)
    # input_hrrpdata = torch.randn(64, 1, 101, 101).to(device)
    # input_textdata = torch.randn(64, 128).to(device)
    # input_list = [input_rcsdata, input_hrrpdata,
    #               input_rcsdata, input_hrrpdata,
    #               input_rcsdata, input_hrrpdata]
    # flops, params = compute_flops_params(model, input_list)

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                 lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)  # weight_decay越大，会进行惩罚，降低变化的速率，一般优化器中的参数都为默认值
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)  # 学习率衰减(指数衰减)，gamma不要设置的太小

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []
    epochs = args.epochs # 多看几轮(13+)
    alpha = args.alpha
    beta = args.beta
    flag = 0

    # 开始训练+测试
    for epoch in range(epochs):

        print("-----第{}轮训练开始------".format(epoch + 1))
        train_loss = 0.0
        test_loss = 0.0
        train_sum, train_cor, test_sum, test_cor = 0, 0, 0, 0
        test_pred = []

        # 训练步骤开始
        model.train()
        for batch_idx, (data_L_rcs, data_L_hrrp, data_S_rcs, data_S_hrrp, data_X_rcs, data_X_hrrp) in tqdm(enumerate(zip(train_rcs_L, train_hrrp_L, train_rcs_S, train_hrrp_S, train_rcs_X, train_hrrp_X)),total=len(train_rcs_L)):
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
            
            optimizer.zero_grad()
            outputs = model(inputs_L_rcs.float(), inputs_L_hrrp.float(), inputs_S_rcs.float(), inputs_S_hrrp.float(), inputs_X_rcs.float(), inputs_X_hrrp.float())  # 需要加.float()，否则会报错
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算每轮训练集的Loss
            train_loss += loss.item()

            # 计算每轮训练集的准确度
            outputs = F.softmax(outputs, dim=1) # 计算准确率时需要添加softmax
            _, predicted = torch.max(outputs.data, 1)  # 选择最大的（概率）值所在的列数就是他所对应的类别数，
            train_cor += (predicted == labels).sum().item()  # 正确分类个数
            train_sum += labels.size(0)  # train_sum+=predicted.shape[0]

            # train_true.append(labels.cpu().numpy()) # 想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。 numpy不能读取CUDA tensor 需要先将它转化为 CPU tensor
            # train_pred.append(predicted.cpu().numpy())
        scheduler.step()

        # 测试步骤开始
        model.eval()
        # with torch.no_grad():
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
            
            optimizer.zero_grad()
            outputs = model(inputs_L_rcs.float(), inputs_L_hrrp.float(), inputs_S_rcs.float(), inputs_S_hrrp.float(), inputs_X_rcs.float(), inputs_X_hrrp.float())  # 需要加.float()，否则会报错
            loss = loss_fn(outputs, labels)
                
            test_loss += loss.item()
            outputs = F.softmax(outputs, dim=1)  # 计算准确率时需要添加softmax
            _, predicted = torch.max(outputs.data, 1)
            test_cor += (predicted == labels).sum().item()
            test_sum += labels.size(0)

            test_pred = test_pred + list(predicted.cpu().numpy())

        print("Train loss:{}   Train accuracy:{}%   Test loss:{}   Test accuracy:{}%".format(train_loss / batch_idx,
                                                                                             100 * train_cor / train_sum,
                                                                                             test_loss / batch_idx1,
                                                                                             100 * test_cor / test_sum))
        train_loss_list.append(train_loss / batch_idx)
        train_acc_list.append(100 * train_cor / train_sum)
        test_acc_list.append(100 * test_cor / test_sum)
        test_loss_list.append(test_loss / batch_idx1)

        # 保存所有损失下的网络
        test_accuracy = 100 * test_cor / test_sum
        if test_accuracy > flag:
            torch.save(model.state_dict(), f"weights/contrast_method/mvfrnet/{snr}dB/model_acc{test_accuracy}.pth")  # model.state_dict保存模型参数、超参数以及优化器的状态信息
            flag = test_accuracy

        # 保存不同损失下的网络
        # if test_accuracy > flag:
        #     torch.save(model.state_dict(), f"weights/Loss_compete/{situation}/{snr}dB/rcs_epoch{epoch}_acc{test_accuracy}.pth")  # model.state_dict保存模型参数、超参数以及优化器的状态信息
        #     flag = test_accuracy

    # 输出混淆矩阵
    # print(len(test_label))
    # print(len(test_pred))
    # print(test_label)
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
    # plt.matshow(C, cmap=plt.cm.Reds)
    # plt.show()

    # 绘制曲线
    # fig = plt.figure()
    # plt.plot(range(len(train_loss_list)), train_loss_list, 'blue')
    # plt.title('TrainLoss', fontsize=14)
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Loss', fontsize=14)
    # plt.grid()
    # plt.savefig(f'result/TrainLoss_RCScnn_{freq}')
    # plt.show()
    # fig = plt.figure()
    # plt.plot(range(len(test_loss_list)), test_loss_list, 'red')
    # plt.title('TestLoss', fontsize=14)
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Loss', fontsize=14)
    # plt.grid()
    # plt.savefig(f'result/TestLoss_RCScnn_{freq}')
    # plt.show()

    # fig = plt.figure()
    # plt.plot(range(len(train_acc_list)), train_acc_list, 'blue')
    # plt.title('TrainAccuracy', fontsize=14)
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Accuracy(%)', fontsize=14)
    # plt.grid()
    # plt.savefig(f'result/TrainAccuracy_RCScnn_{freq}')
    # plt.show()
    # fig = plt.figure()
    # plt.plot(range(len(test_acc_list)), test_acc_list, 'red')
    # plt.title('TestAccuracy', fontsize=14)
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Accuracy(%)', fontsize=14)
    # plt.grid()
    # plt.savefig(f'result/TestAccuracy_RCScnn_{freq}')
    # plt.show()