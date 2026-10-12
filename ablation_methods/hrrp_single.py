'''
    使用时移至最外层
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
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from mamba_ssm import Mamba

from utils.data_utils import Dataset
from model.ablation_model import Mamba_Net_ablation


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")
    # 初始化参数
    snr = 0
    freq = 'X'
    batch_size = 64

    dataset = Dataset(data_path=f'dataset/HRRP_data_{freq}_{snr}dB.mat', batch_size=batch_size)
    train_hrrp = dataset.create_dataloader(type='train')
    test_hrrp = dataset.create_dataloader(type='test')
    test_label = dataset.get_test_label()

    model = Mamba_Net_ablation(d_model=101, output_dim=4)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    # 注意，1.如果用SGD优化器，一般学习率大一点，如0.1、0.05；Adam优化器学习率可以小一点,如0.001、0.0001
    #      2.最好加上weight_decay(L2正则化)和学习率衰减
    # optimizer=torch.optim.SGD(params=model.parameters(),lr=learning_rate,weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                 lr=0.001, betas=(0.9, 0.99), weight_decay=1e-4)  # weight_decay越大，会进行惩罚，降低变化的速率，一般优化器中的参数都为默认值
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)  # 学习率衰减(指数衰减)，gamma不要设置的太小

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []
    flag = 0
    epochs = 50

    # 开始训练+测试
    for epoch in range(epochs):

        print("-----第{}轮训练开始------".format(epoch + 1))
        train_loss = 0.0
        test_loss = 0.0
        train_sum, train_cor, test_sum, test_cor = 0, 0, 0, 0
        # arr=np.zeros((1,3630))
        test_pred = []

        # 训练步骤开始
        model.train()
        for batch_idx, data in tqdm(enumerate(train_hrrp),total=len(train_hrrp)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # labels = torch.tensor(labels, dtype=torch.long)  # 需要将label转换成long类型
            labels = labels.clone().detach()
            optimizer.zero_grad()
            outputs = model(inputs.float())  # 需要加.float()，否则会报错
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算每轮训练集的Loss
            train_loss += loss.item()

            # 计算每轮训练集的准确度
            outputs = F.softmax(outputs, dim=1)  # 计算准确率时需要添加softmax
            _, predicted = torch.max(outputs.data, 1)  # 选择最大的（概率）值所在的列数就是他所对应的类别数，
            train_cor += (predicted == labels).sum().item()  # 正确分类个数
            train_sum += labels.size(0)  # train_sum+=predicted.shape[0]

            # train_true.append(labels.cpu().numpy()) #想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。 numpy不能读取CUDA tensor 需要先将它转化为 CPU tensor
            # train_pred.append(predicted.cpu().numpy())
        scheduler.step()

        # 测试步骤开始
        model.eval()
        # with torch.no_grad():
        for batch_idx1, data in tqdm(enumerate(test_hrrp),total=len(test_hrrp)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.clone().detach()

            outputs = model(inputs.float())
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

        # 保存网络
        test_accuracy = 100 * test_cor / test_sum
        if test_accuracy > flag:
            torch.save(model.state_dict(), f"weights/ablation_method/hrrp_single/{freq}/{snr}dB/model_acc{test_accuracy}.pth")  # model.state_dict保存模型参数、超参数以及优化器的状态信息
            flag = test_accuracy

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
