%% 获取进动俯仰角变化曲线  
clear;clc;
% 输入想要得到的频段
freq = input('Input the frequency:','s');
% 为不同频段设置初始俯仰角
phi_P = 31:1:43;
phi_C = 31:1:43;
phi_L = 31:1:43;
phi_S = 21:1:33;
phi_X = 41:1:53;

% 其他参数设置
PRF=20; %脉冲重复频率20
T=5; % 观测时间
phi = eval(sprintf('phi_%s',freq));
fc=2:0.2:4; %进动频率
alpha_c=2:0.15:3.5; %进动角    每个目标共13×11×11=1573个数据
fs=3;  %自旋
t_max=T;

%验证 
% [z]=chouqujiaodu_600(3,fs,2,T,20,600);
% figure;plot(z);

aa=1;
for kk=1:length(phi)
    for ii=1:length(fc)
        for jj=1:length(alpha_c)
            angle(aa,:)=chouqujiaodu_jindong_600(alpha_c(jj),fs,fc(ii),t_max,phi(kk),PRF); %
            aa=aa+1;
        end
    end
end
angle_target=angle;
save(['angle_target44_',freq,'.mat'],'angle_target');
% save angle_target1  angle_target; %第一个是保存的数据的名字；第二个是要保存的数据

