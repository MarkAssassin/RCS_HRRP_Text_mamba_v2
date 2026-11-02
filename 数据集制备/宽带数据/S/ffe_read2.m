%% 从 ffe 文件中读取回波及 RCS，并降带宽到200MHz
clc; clear; close all;

%% =========================
% 1. 读取 FFE 文件
% =========================
fidin = fopen('Test4_FarField1_S.ffe'); % 1-4的顺序即为论文中目标1-4的顺序
fidout = fopen('mk420.txt','w');

% 跳过前6行头文件
for k = 1:6
    tline = fgetl(fidin);
end

% 逐行读取数据
while ~feof(fidin)
    for k = 1:12   % Linux:12 / Windows:10
        tline = fgetl(fidin);
    end

    fprintf(fidout,'%s',tline);
    fprintf(fidout,'%s',char(13,10)');
    continue;
end

fclose(fidin);
fclose(fidout);

%% =========================
% 2. 读取数据
% =========================
mk = importdata('mk420.txt');

% 复数回波（水平极化 5,6列）
A = mk(:,5) + 1j * mk(:,6);

%% =========================
% 3. 重构频率-角度矩阵
% =========================
data = reshape(A,[1801,101]);  % 1801角度 × 101频点
data = data.';                 % 101 × 1801

%% =========================
%% =========================
% 4. 200MHz带宽处理（保持101维）
% =========================
BW_total = 2e9; % 2GHz
BW_new   = 6e8; % 600MHz

N_total = size(data,1);   % 101
df = BW_total / (N_total-1); % 20 MHz

% 300MHz对应频点数
N_new = round(BW_new / df) + 1;   % = 16

% 取中心频段
center_idx = floor(N_total/2) + 1;
half = floor(N_new/2);

idx_band = (center_idx-half):(center_idx+half);

% 防止越界（重要）
idx_band = max(idx_band,1);
idx_band = min(idx_band,N_total);

% =========================
% Step 1：提取300MHz子带
% =========================
data_band = data(idx_band, :);   % 16 × 3601

% =========================
% Step 2：插值回101维（保持结构不变）
% =========================
x_old = linspace(0,1,size(data_band,1));
x_new = linspace(0,1,101);

data = interp1(x_old, data_band, x_new, 'spline');

%% =========================
% 5. 转回角度维为主维
% =========================
% data = data.';   % 1801 × 101

%% =========================
% 6. 构造360°RCS（左右镜像拼接）
% =========================
data2 = fliplr(data(:,1:1800));
data3 = [data, data2];

data = data3;

save('qiudizhuizhuti_600MHz.mat','data');

%% =========================
% 7. RCS可视化
% =========================
figure;
plot(0:0.1:360, 10*log(abs(data3)+1e-12));
xlabel('俯仰角（°）','fontsize',14);
ylabel('RCS（dB）','fontsize',14);
set(gca,'XTick',[0:40:360]);
set(gca,'XLim',[0,360]);

%% =========================
% 8. HRRP生成
% =========================
data_hrrp = fftshift(ifft(data,[],1),1);

figure;
colormap hot;
imagesc(abs(data_hrrp).^0.2);
xlabel('频点/距离单元');
ylabel('角度');
title('HRRP (200 MHz Band)');