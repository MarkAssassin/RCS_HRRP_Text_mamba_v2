%% 从ffe文件中读取回波及RCS的程序代码
% 输入：目标ffe文件
% 输出：目标静态回波数据及RCS数据
clc;clear all;
close all;
fidin=fopen('Test4_FarField1_S.ffe'); % 打开目标1的ffe文件
fidout=fopen('mk420.txt','w'); % 创建mk420空白文件，以便后续写入
for k = 1:6
    tline = fgetl(fidin); % 返回ffe文件中的下一行，并删除换行符
end
while ~feof(fidin) % 若未读到ffe文件末尾，则继续循环
    for k = 1:12     % linux系统产生的数据  k = 1：12  ；  windows下k = 1:10；
        tline=fgetl(fidin); % 返回ffe文件中的下一行，并删除换行符
    end
        fprintf(fidout,'%s',tline);
        fprintf(fidout,'%s',char(13,10)');% 10换行   13回车
    continue;
end
fclose(fidout); % 停止写入mk420文件
mk=importdata('mk420.txt');% 将mk420.txt加载到数组mk中
%% 若要导出静态回波文件
A=mk(:,5)+1j*mk(:,6);   %第3列为实部，第4列为虚部(3 4为垂直接收，5 6为水平接收)
%% 若要导入静态RCS文件
% A=mk(:,9);             %第9列为RCS
%% 保存文件
data = reshape(A,[1801,101]); % 1801表示1801个角度间隔，2表示频点数
% data=data(:,1);

data=data.'; % 共轭转置
% save yuanzhui_rcs data; % 保存所读取的文件，yuanzhui_rcs为文件名

data2=fliplr(data(:,1:1800)); %反转数组，因为在电磁仿真时俯仰角从0-180，没到360，这是因为目标是对称的，只要在这里翻转再拼接即可
data3=[data,data2]; %注意，如果是多个频点(比如101)，需要修改为data2=fliplr(data(:,1:1800));
data=data3;
save('qiudizhuizhuti.mat','data');
% save target1 target1;
figure;plot(0:0.1:360,10*log(abs(data3)));
xlabel('俯仰角（°）','fontsize',14);ylabel('RCS（dB）','fontsize',14);
set(gca,'XTick',[0:40:360]);
set(gca,'XLim',[0,360]); 
% title('球顶锥体静态RCS曲线(窄带)','fontsize',14);
% saveas(gcf,'Test1_RCS曲线_窄带','png');

%%
clear; clc;

data = load('qiudizhuizhuti_600MHz.mat').data;
data_hrrp = fftshift(ifft(data, [], 1), 1);

img = abs(data_hrrp).^0.2;

figure;
colormap hot;

% =========================
% 关键：设置坐标范围
% =========================
[nRange, nAngle] = size(img);

imagesc([0 360], [0 100], img);   % 横轴0-360，纵轴0-100
% set(gca, 'YDir', 'normal');       % 保证纵轴向上递增

xlabel('Elevation angle (°)', 'fontsize', 14);
ylabel('Range bins', 'fontsize', 14);

% =========================
% 设置刻度（重点）
% =========================
xticks(0:60:360);
yticks(0:20:100);
