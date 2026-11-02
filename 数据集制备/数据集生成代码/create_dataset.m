%% 构建数据集
clear;clc;
% 训练&测试样本数
train_num = 10*11*11;
test_num = 3*11*11;
snr = 5;
% 下载rcs数据
freq_list = ['P' 'L' 'S' 'X' 'C'];
freq = input('Input the frequency:','s');
if any(freq == freq_list)
    disp('Right frequency!');
else
    error('No such frequency!');
end
rcs_data = load(['rcs_',freq,'.mat']).rcs_zhaidai_cell;
% hrrp_data = load(['hrrp_',freq,'.mat']).hrrp_kuandai_cell;

% 绘制RCS时域图像
% 时域
% for i=1:4
%     figure;
%     plot(abs(rcs_data{i}(1500,:)));
%     set(gca,'XLim',[0,101]);
% end 

%% 搭建RCS数据集 1573×101
% 训练集数据
train_data = [];
for i=1:length(rcs_data)
    target_data = rcs_data{i}(1:train_num,:);
    train_data = [train_data;target_data];
end
% 添加噪声
train_data = awgn(train_data,snr,'measured');
save(['rcs_train_data_',freq,'_',num2str(snr),'dB'],'train_data');
% 训练集标签
train_labels=zeros(1,train_num*4);
train_labels(1,1:train_num)=0;
train_labels(1,train_num+1:train_num*2)=1;
train_labels(1,train_num*2+1:train_num*3)=2;
train_labels(1,train_num*3+1:train_num*4)=3;

% 搭建测试集
test_data = [];
for i=1:length(rcs_data)
    target_data = rcs_data{i}(train_num+1:end,:);
    test_data = [test_data;target_data];
end
% 添加噪声
test_data = awgn(test_data,snr,'measured');
save(['rcs_test_data_',freq,'_',num2str(snr),'dB'],'test_data');
% 测试集标签
test_labels=zeros(1,test_num*4);
test_labels(1,1:test_num)=0;
test_labels(1,test_num+1:test_num*2)=1;
test_labels(1,test_num*2+1:test_num*3)=2;
test_labels(1,test_num*3+1:test_num*4)=3;

%保存数据集及标签
save(['RCS_data_',freq,'_',num2str(snr),'dB','.mat'],'test_data','test_labels','train_data','train_labels');

%% 搭建HRRP数据集  101×101×1573
% 训练集数据
train_data = [];
for i=1:length(hrrp_data)
    target_data = hrrp_data{i}(:,:,1:train_num);
    train_data = cat(3,train_data,target_data);
end
% 添加噪声
train_data = awgn(train_data,snr,'measured');
save(['hrrp_train_data_',freq,'_',num2str(snr),'dB'],'train_data');
% 训练集标签
train_labels=zeros(1,train_num*4);
train_labels(1,1:train_num)=0;
train_labels(1,train_num+1:train_num*2)=1;
train_labels(1,train_num*2+1:train_num*3)=2;
train_labels(1,train_num*3+1:train_num*4)=3;

% 搭建测试集
test_data = [];
for i=1:length(hrrp_data)
    target_data = hrrp_data{i}(:,:,train_num+1:end);
    test_data = cat(3,test_data,target_data);
end
% 添加噪声
test_data = awgn(test_data,snr,'measured');
save(['hrrp_test_data_',freq,'_',num2str(snr),'dB'],'test_data');
% 测试集标签
test_labels=zeros(1,test_num*4);
test_labels(1,1:test_num)=0;
test_labels(1,test_num+1:test_num*2)=1;
test_labels(1,test_num*2+1:test_num*3)=2;
test_labels(1,test_num*3+1:test_num*4)=3;

%保存数据集及标签
save(['HRRP_data_',freq,'_',num2str(snr),'dB','.mat'],'test_data','test_labels','train_data','train_labels');

%% RCS数据
data_1 = rcs_zhaidai_cell{1};
data_2 = rcs_zhaidai_cell{2};
data_3 = rcs_zhaidai_cell{3};
data_4 = rcs_zhaidai_cell{4};

%在当前工作目录下新建文件夹并保存
mkdir image_rcs/target1/ % 如果文件夹已存在，会有警告，但不影响运行
figure;
for i=1:length(data_1)
    plot(abs(data_1(i,:)));
    saveas(gca,strcat('image_rcs/target1/',num2str(i),'.png'));
    pause(0.1);
end

% figure;
% for i=1:length(data_1)
%     subplot(2,2,1);
%     plot(abs(data_1(i,:)));
%     subplot(2,2,2);
%     plot(abs(data_2(i,:)));
%     subplot(2,2,3);
%     plot(abs(data_3(i,:)));
%     subplot(2,2,4);
%     plot(abs(data_4(i,:)));
%     pause(0.5);
% end

%% HRRP数据
data_1 = hrrp_kuandai_cell{1};
data_2 = hrrp_kuandai_cell{2};
data_3 = hrrp_kuandai_cell{3};
data_4 = hrrp_kuandai_cell{4};
figure;
for i=1:size(data_1,3)
    subplot(2,2,1);
    imagesc(abs(data_1(:,:,i)));
    subplot(2,2,2);
    imagesc(abs(data_2(:,:,i)));
    subplot(2,2,3);
    imagesc(abs(data_3(:,:,i)));
    subplot(2,2,4);
    imagesc(abs(data_4(:,:,i)));
    pause(0.1);
end

mkdir image_hrrp
figure;
imagesc(abs(data_4(:,:,1)));
saveas(gca,strcat('image_hrrp/4.png'));

