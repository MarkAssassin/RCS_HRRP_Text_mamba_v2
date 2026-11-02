function  Rt=zixuanjvzhen(l,f,t)
%l是旋转所绕单位方向矢量，为列向量
%f瑟吉欧旋转角速度
%t是时间
%R是自旋矩阵
e=[0,-l(3,1),l(2,1);l(3,1),0,-l(1,1);-l(2,1),l(1,1),0];
Rt=[];
for i=1:1:size(t,1)
Rt(:,:,i)=eye(3)+e*sin(f*t(i,1))+e*e*(1-cos(f*t(i,1)));
end