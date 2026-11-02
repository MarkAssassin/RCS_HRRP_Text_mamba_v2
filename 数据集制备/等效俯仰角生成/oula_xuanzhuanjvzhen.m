function  Rinit=oula_xuanzhuanjvzhen(oulajiao)
%oulajiao为欧拉角，单位是弧度
a=oulajiao(1,1);
b=oulajiao(2,1);
c=oulajiao(3,1);
Ra=[cos(a),-sin(a),0;sin(a),cos(a),0;0,0,1];
Rb=[1,0,0;0,cos(b),-sin(b);0,sin(b),cos(b)];
Rc=[cos(c),-sin(c),0;sin(c),cos(c),0;0,0,1];
Rinit=Ra*Rb*Rc;