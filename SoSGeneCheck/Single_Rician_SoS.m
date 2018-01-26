%%%%%%%%----------------���������SOSģ�͵���˹�ŵ�------------------%%%%%%%%
function [H] = Single_Rician_SoS(fs,f_dopp,AOA,Rice_K,endT)
%%---- AOA�����ŵ��Ķ�����Ƶ�Ʋ���  Rice_K ��˹˥������----%%
%%---- PSD�������չ�������״                          ----%%
H = []; M = 128; frameT = 1024/fs;                                               
while endT > 1/fs
    if endT > frameT
        len = frameT;
        endT = endT - frameT;
    else 
        len = round(endT*fs)/fs;
        endT = 0;
    end
    rand('state',sum(100*clock));         
    fic = unifrnd(-pi,pi);  fis = unifrnd(-pi,pi);                      
    %afi = 2*pi*[1:M]/M + unifrnd(-pi,pi,1,M)/M;                         %%%NLOS�����������������ַ���
    afi = ( 2*pi*[1:M]-pi+unifrnd(-pi,pi) )/( 4*M );               
    dafic(1:M) = unifrnd(-pi,pi,1,M);dafis(1:M) = unifrnd(-pi,pi,1,M);  
    tmp1 = sqrt(1/M);tmp2 = sqrt(Rice_K);
    tmp3 = 2*pi*f_dopp*cos(afi);tmp4 = 2*pi*f_dopp*sin(afi);
    tmp5 = 2*pi*f_dopp*cos(AOA);tmp6 = 2*pi*f_dopp*cos(AOA);             %%%LOS����������ǹ̶�
    tmp7 = sqrt(Rice_K+1);
    nTs = 0;
    for t = 1/fs : 1/fs : len
        nTs = nTs + 1;
        Hc(nTs) = single( tmp1*sum(cos(tmp3*t+dafic)) + tmp2*cos(tmp5*t+fic) )/tmp7;
        Hs(nTs) = single( tmp1*sum(sin(tmp4*t+dafic)) + tmp2*sin(tmp6*t+fic) )/tmp7;
    end     
    H = [H Hc + i*Hs];
    clear Hc;clear Hs;
end





