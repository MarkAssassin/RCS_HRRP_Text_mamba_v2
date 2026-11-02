%% 读取宽带P\S\X频段数据
% L\S\X载频分别为1.5GHz、3GHz和10GHz,宽带带宽设置为2GHz
clear;clc;
% 输入想要得到的频段
freq = input('Input the frequency:','s');
% freq_list = ['P' 'L' 'S' 'X' 'C'];
freq_list = ['S' 'C' 'X'];
if any(freq == freq_list)
    disp('Right frequency!');
else
    error('No such frequency!');
end
file_path = ['C:\Users\86178\Desktop\data_creator\宽带数据\',freq,'\'];
if freq == 'X'
    data_target1 = load([file_path,'dayuanzhui.mat']).feko_data;
    data_target2 = load([file_path,'zhuizhuti.mat']).feko_data;
    data_target3 = load([file_path,'qiudizhui.mat']).feko_data;
    data_target4 = load([file_path,'qiudizhuizhuti.mat']).feko_data;
elseif freq == 'S'
    data_target1 = load([file_path,'dayuanzhui_600MHz.mat']).data;
    data_target2 = load([file_path,'zhuizhuti_600MHz.mat']).data;
    data_target3 = load([file_path,'qiudizhui_600MHz.mat']).data;
    data_target4 = load([file_path,'qiudizhuizhuti_600MHz.mat']).data;
elseif freq == 'C'
    data_target1 = load([file_path,'dayuanzhui_800MHz.mat']).data;
    data_target2 = load([file_path,'zhuizhuti_800MHz.mat']).data;
    data_target3 = load([file_path,'qiudizhui_800MHz.mat']).data;
    data_target4 = load([file_path,'qiudizhuizhuti_800MHz.mat']).data;
end
% 将4类目标的rcs数据保存在一个cell中
data_cell = cell(1,4);
for i = 1:4
    data_cell{i} = eval(sprintf('data_target%d',i));
end

% 目标1-4的等效俯仰角  维度:400×1201
anle_path = 'C:\Users\86178\Desktop\data_creator\等效俯仰角生成\';
angle_target1 = load([anle_path,'angle_target1_',freq,'.mat']).angle_target;
angle_target2 = load([anle_path,'angle_target2_',freq,'.mat']).angle_target;
angle_target3 = load([anle_path,'angle_target3_',freq,'.mat']).angle_target;
angle_target4 = load([anle_path,'angle_target4_',freq,'.mat']).angle_target;
angle_cell = cell(1,4);
for i = 1:4
    angle_cell{i} = eval(sprintf('angle_target%d',i));
end

% 制作HRRP数据
% 动态抽取
PRF=20;
T=5;
t=0:1/PRF:T;
snr = 20;
hrrp_cell = cell(1,4);
for i=1:4
    data = data_cell{i};
    angle = angle_cell{i};
    for num=1:size(angle_cell{i},1) %等效俯仰角序列共1573个
        for j=1:size(data_cell{i},1) %101个频点
            for k=1:length(t) %采样点数为101
                data_chouqu(j,k,num) = data(j,roundn(angle(num,k),-1)*10+1); %roundn(phai_equivalent(1,i),-1)*10+1表示等效俯仰角在原始回波信号中的序列号
            end
        end
    end
    data_chouqu = awgn(data_chouqu,snr,'measured'); % 先加噪，再脉压
    hrrp_cell{i} = data_chouqu;
end
% 对回波数据进行脉冲压缩，得到hrrp数据
hrrp_kuandai_cell = cell(1,4);
for i=1:4
    for num=1:size(hrrp_cell{i},3)
        hrrp_kuandai_cell{i}(:,:,num) = fftshift(ifft(hrrp_cell{i}(:,:,num),[],1),1);
    end
end
figure;
imagesc(abs(hrrp_kuandai_cell{2}(:,:,500)));

% 保存数据
% save(['hrrp_',freq],'hrrp_kuandai_cell')  % 列数代表目标序号
%% 构建数据集
% 训练&测试样本数
train_num = 10*11*11;
test_num = 3*11*11;

%% 搭建HRRP数据集  101×101×1573
% 训练集数据
train_data = [];
for i=1:length(hrrp_kuandai_cell)
    target_data = hrrp_kuandai_cell{i}(:,:,1:train_num);
    train_data = cat(3,train_data,target_data);
end
% save(['hrrp_train_data_',freq,'_',num2str(snr),'dB'],'train_data');
% 训练集标签
train_labels=zeros(1,train_num*4);
train_labels(1,1:train_num)=0;
train_labels(1,train_num+1:train_num*2)=1;
train_labels(1,train_num*2+1:train_num*3)=2;
train_labels(1,train_num*3+1:train_num*4)=3;

% 搭建测试集
test_data = [];
for i=1:length(hrrp_kuandai_cell)
    target_data = hrrp_kuandai_cell{i}(:,:,train_num+1:end);
    test_data = cat(3,test_data,target_data);
end
% save(['hrrp_test_data_',freq,'_',num2str(snr),'dB'],'test_data');
% 测试集标签
test_labels=zeros(1,test_num*4);
test_labels(1,1:test_num)=0;
test_labels(1,test_num+1:test_num*2)=1;
test_labels(1,test_num*2+1:test_num*3)=2;
test_labels(1,test_num*3+1:test_num*4)=3;

%保存数据集及标签
save(['HRRP_data_',freq,'_',num2str(snr),'dB','.mat'],'test_data','test_labels','train_data','train_labels');
