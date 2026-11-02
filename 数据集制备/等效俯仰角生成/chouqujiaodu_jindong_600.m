function  [z]=chouqujiaodu_jindong_600(jindongjiao,fs,fc,t_max,chushifuyangjiao,prf)
%% 输入进动角角度，fs 自旋频率   fc 进动频率
jindongjiao1=jindongjiao*pi/180;
chushifuyangjiao1=chushifuyangjiao*pi/180;
%  R0=[15;30;70]*1e3; 

Target = [0  1  0  % y轴
          0  0   1 % z轴
          1  0   0 ]; % x轴
oulajiao=[50;20;45]/180*pi;%目标系转到参考系的欧拉角
% oulajiao_1=oulajiao(1);oulajiao_2=oulajiao(2);oulajiao_3=oulajiao(3);
Rinit=oula_xuanzhuanjvzhen(oulajiao);%坐标转换矩阵
lz_mubiaoxi=[0;0;1];%lz是自旋轴
lz_cankaoxi=Rinit*lz_mubiaoxi;
syms z1 z2
% lc_cankaoxi=[0.1;z2;z1];
[z1,z2]=...
vpasolve((dot(lz_cankaoxi,[0.1;z2;z1])/(norm(lz_cankaoxi)*norm([0.1;z2;z1]))-cos(chushifuyangjiao1)),[z1,z2]);

z2=double(z2);
z1=double(z1);
R00=[0.1;z2;z1];
R0=R00./norm(R00);

%%
syms z1 z2
% lc_cankaoxi=[0.1;z2;z1];
[z1,z2]=...
vpasolve((dot(lz_cankaoxi,[0.1;z2;z1])/(norm(lz_cankaoxi)*norm([0.1;z2;z1]))-cos(jindongjiao1)),[z1,z2]); %vpasolve函数可以对数值解方程进行求解

z2=double(z2);
z1=double(z1);
lc1=[0.1;z2;z1];
lc_cankaoxi=lc1./norm(lc1); % 根据进动角计算出来进动轴的单位方向矢量 
% jindongjiao1=acos(dot(lc_cankaoxi,lz_cankaoxi)/(norm(lc_cankaoxi)*norm(lz_cankaoxi)))

% los_jindongzhou=acos(dot(lc_cankaoxi,R0)/(norm(lc_cankaoxi)*norm(R0)))*180/pi;

PRF=prf;  
t=(0:1/PRF:t_max).';%时间
f1=2*pi*fs;%自旋频率2Hz
Rs=zixuanjvzhen(lz_cankaoxi,f1,t);%  自旋矩阵
f2=2*pi*fc; % 锥旋频率2Hz;
Rc=zixuanjvzhen(lc_cankaoxi,f2,t);  % 进动矩阵
for ii=1:1:size(t,1)
       Rt(:,:,ii)=Rc(:,:,ii)*Rs(:,:,ii);
%      Rt(:,:,ii)=Rs(:,:,ii);
  %    Rt(:,:,ii)=Rc(:,:,ii);
end                                    %脉冲重复频率
Target=Target.';
for i=1:1:size(Target,2)
    for ii=1:1:size(Rt,3)
        r=Rt(:,:,ii)*oula_xuanzhuanjvzhen(oulajiao)*Target(:,i); %参考坐标系中的位置矢量，r=微动矩阵(Rt)*欧拉旋转矩阵(Rinit)*散射点在本地坐标系的初始位置
        r2(:,ii)=r;
%           r2(:,ii)=(r);    
    end   
    rr2(:,:,i)=r2;
end
rr2;% 三个点每时每刻在目标坐标系中的坐标 
qz=rr2(:,:,2);% 计算俯仰角用到的矢量  z
qx=rr2(:,:,3);%  计算方位角用到的矢量 x
qy=rr2(:,:,1);% 计算方位角用到的矢量y

% for i=1:size(qz,2)
%     q2222(:,i)=qx(:,i)'*qy(:,i)       %验证是否垂直的程序  17号晚上的程序是错误的  现已证明垂直
%     q1111(:,i)=qx(:,i)'*qz(:,i)  
%     q3333(:,i)=qy(:,i)'*qz(:,i)  
% end


LOS=R0;% 雷达视线矢量
% u1=zeros(size(qx));
% u2=zeros(size(qx));
q4=zeros(size(qx));

% for i=1:size(qx,2)
% u1=(dot(qx(:,i),LOS)/(norm(qx(:,i))*norm(LOS))).*qx0;
% u2=(dot(qy(:,i),LOS)/(norm(qy(:,i))*norm(LOS))).*qy0;
% q4=u1+u2;
% end
% 将LOS投影到新建坐标系的（x,y）面投影向量用q4表示
for i=1:size(qx,2)
    n=cross(qx(:,i),qy(:,i)); %叉积
    n1=n./norm(n);
    h=dot(LOS,n1);
    P=n1.*h;
    q4(:,i)=LOS-P; 
end
for i=1:size(qz,2)
    fuyangjiao(1,i)=acos(dot(qz(:,i),LOS)/(norm(qz(:,i))*norm(LOS)))*180/pi;
end
for i=1:size(qz,2)
    fangweijiao1(1,i)=acos(dot(q4(:,i),qx(:,i))/(norm(q4(:,i))*norm(qx(:,i))))*180/pi;
    fangweijiao2(1,i)=acos(dot(q4(:,i),qy(:,i))/(norm(q4(:,i))*norm(qy(:,i))))*180/pi;
    if fangweijiao2(1,i)>90
        fangweijiao1(1,i)=360-fangweijiao1(1,i);
    else
        fangweijiao1(1,i)=fangweijiao1(1,i);
    end
end
%  figure;plot(fuyangjiao);
% % 
%  figure;plot(fangweijiao1);

z0=cat(1,fangweijiao1,fuyangjiao); 
z=z0(2,:);


