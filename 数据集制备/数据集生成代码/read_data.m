%% 读取窄带P\S\X频段数据
% P\L\S\C\X载频分别为500MHz、1.5GHz、3GHz、6GHz和10GHz,窄带带宽设置为20MHz
% 不同频段初始俯仰角也不同
% phi_P = 31:1:43;
% phi_S = 21:1:33;
% phi_X = 41:1:53;
clear;clc;
% 输入想要得到的频段
freq = input('Input the frequency:','s');
% freq_list = ['P' 'L' 'S' 'X' 'C'];
freq_list = ['C' 'S' 'X'];
if any(freq == freq_list)
    disp('Right frequency!');
else
    error('No such frequency!');
end
file_path = ['D:\研一\论文\RCS+HRRP图网络融合识别\data\narrow-band\',freq,'\'];
data_target1 = load([file_path,'dayuanzhui.mat']).data;
data_target2 = load([file_path,'zhuizhuti.mat']).data;
data_target3 = load([file_path,'qiudizhui.mat']).data;
data_target4 = load([file_path,'qiudizhuizhuti.mat']).data;

% 将4类目标的rcs数据保存在一个cell中
data_cell = cell(1,4);
for i = 1:4
    data_cell{i} = eval(sprintf('data_target%d',i));
end

% 静态RCS
% data2=fliplr(data_target1_X(51,1:1800)); %反转数组，因为在电磁仿真时俯仰角从0-180，没到360，这是因为目标是对称的，只要在这里翻转再拼接即可
% data3=[data_target1(51,:),data2]; 
% figure;
% plot(0:0.1:360,10*log(abs(data3)));
% set(gca,'XTick',[0:40:360]);  
% set(gca,'XLim',[0,360]);
% xlabel('俯仰角（°）','fontsize',14);ylabel('RCS（dB）','fontsize',14);

% 目标1-4的等效俯仰角  维度:400×1201
anle_path = 'D:\研一\论文\RCS+HRRP图网络融合识别\data\angle\';
angle_target1 = load([anle_path,'angle_target1_',freq,'.mat']).angle_target;
angle_target2 = load([anle_path,'angle_target2_',freq,'.mat']).angle_target;
angle_target3 = load([anle_path,'angle_target3_',freq,'.mat']).angle_target;
angle_target4 = load([anle_path,'angle_target4_',freq,'.mat']).angle_target;

angle_cell = cell(1,4);
for i = 1:4
    angle_cell{i} = eval(sprintf('angle_target%d',i));
end

% 制作RCS数据
% 动态抽取
PRF=20;
T=5;
t=0:1/PRF:T;

if freq == 'X'
    rcs_cell = cell(1,4);
    for i = 1:4
        data = data_cell{i};
        angle = angle_cell{i};
        for num=1:size(angle_cell{i},1) %等效俯仰角序列共1573个
            for j=1:size(data_cell{i},1) %101个频点
                for k=1:length(t) %采样点数为101
                    data_chouqu(j,k,num) = data(j,roundn(angle(num,k),-1)*10+1); %roundn(phai_equivalent(1,i),-1)*10+1表示等效俯仰角在原始回波信号中的序列号
                end
            end
        end
        rcs_cell{i} = data_chouqu;
    end
    % 对rcs数据进行脉压抽取得到最终的窄带rcs数据(对X频段，因为生成X频段数据时有101个频点，而S\P只有2个)
    rcs_zhaidai_cell = cell(1,4);
    for i=1:4
        for num=1:size(rcs_cell{i},3)
            rcs_zhaidai(:,:,num) = fftshift(ifft(rcs_cell{i}(:,:,num),[],1),1);
            rcs_zhaidai_chouqu(num,:) = rcs_zhaidai(51,:,num);
        end
        rcs_zhaidai_cell{i} = rcs_zhaidai_chouqu;
    end

elseif freq == 'S' || freq == 'P' || freq == 'L' || freq == 'C'
    rcs_cell = cell(1,4);
    rcs_zhaidai_cell = cell(1,4);
    for i = 1:4
        data = data_cell{i};
        angle = angle_cell{i};
        for num=1:size(angle_cell{i},1) %等效俯仰角序列共400个
            for j=1:length(t) %采样点数为101
                data_chouqu(num,j) = data(1,roundn(angle(num,j),-1)*10+1); %roundn(phai_equivalent(1,i),-1)*10+1表示等效俯仰角在原始回波信号中的序列号
            end
        end
    rcs_cell{i} = data_chouqu;
    end
    rcs_zhaidai_cell = rcs_cell;
end

% figure;
% imagesc(abs(rcs_zhaidai_cell{1}(:,:,1)));
% xlabel('脉冲序列','fontsize',14);ylabel('距离向（m）','fontsize',14);
% figure;
% plot(abs(rcs_zhaidai_cell{3}(1,:)));

save(['rcs_',freq],'rcs_zhaidai_cell')  % 列数代表目标序号

%% 读取宽带P\S\X频段数据
% L\S\X载频分别为1.5GHz、3GHz和10GHz,宽带带宽设置为2GHz
clear;clc;
% 输入想要得到的频段
freq = input('Input the frequency:','s');
% freq_list = ['P' 'L' 'S' 'X' 'C'];
freq_list = ['L' 'S' 'C' 'X'];
if any(freq == freq_list)
    disp('Right frequency!');
else
    error('No such frequency!');
end
file_path = ['D:\研一\论文\RCS+HRRP图网络融合识别\data\broad-band\',freq,'\'];
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
anle_path = 'D:\研一\论文\RCS+HRRP图网络融合识别\data\angle\';
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
save(['hrrp_',freq],'hrrp_kuandai_cell')  % 列数代表目标序号