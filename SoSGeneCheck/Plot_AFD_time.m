%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AFD--电平持续时间统计曲线,信道包络低于某电平的时间
% H  输入信道矩阵，行数-信道数目，列数-信道持续点数 ；
% fs 采样率
% dB_range 统计参考电平的范围,相对包络均方根的比值；
% fd 最大多普勒频率，求瑞利信道理论AFD时需要；
% Plot_flag '2'-画图,并与瑞利信道理论值比较 ‘1’-只画图  '0'-不画图
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [levelt] = Plot_AFD_time(H, fs, dB_range, fd, Nak_m, Plot_flag)

if nargin == 5
    Plot_flag = 0;
end

% close all;clear all; clc;
% fd = 200;
% fs = 8000;
% H = Single_Rayleigh_IDFT(fs,fd,10000/fs,0);
% dB_range = [-30:2:10];
% Plot_flag = 2;

rou_rms = 10.^(dB_range/20);  %%线性比值 rou_rms^2 = level^2 / Power
line_n = size(H,1); row_n = size(H,2);
if line_n >= row_n
    disp('H is error!')
    return;
end
for method = 1:line_n
    nnn = 0;
    for level = sqrt( (rou_rms.^2)*mean(abs(H(method,:)).^2) )  %% 比较电平，相对包络均方根的值
        nnn = nnn+1;
        lowflag = 0;
        levelt(method,nnn) = 0;
        nn = 0;
        for n=1:row_n
            if n == 1 && abs(H(method,n+1))<=level
               lowflag = 1;
               tempt = 1;
               nn = nn+1; 
            elseif n == row_n
                if lowflag==1
                    levelt(method,nnn) = levelt(method,nnn) + tempt; 
                end
            elseif abs(H(method,n))>=level && abs(H(method,n+1))<level && lowflag==0
               lowflag = 1;
               tempt = 1;
               nn = nn+1;
            elseif abs(H(method,n))<level && abs(H(method,n+1))<level && lowflag==1
               tempt = tempt+1;
            elseif abs(H(method,n))<level && abs(H(method,n+1))>=level && lowflag==1
               levelt(method,nnn) = levelt(method,nnn) + tempt;
               tempt = 0;
               lowflag = 0;
            end
        end
        levelt(method,nnn) = levelt(method,nnn)/max(1,nn); %% 持续时间求平均
    end
    levelt(method,:) = levelt(method,:)/fs; %% 持续时间(s)/每个电平
end

if Plot_flag == 2
    figure;
    AFD = (exp(rou_rms.^2)-1)./(rou_rms.*fd*sqrt(2*pi));
    % AFD = gammainc(Nak_m*rou_rms.^2,Nak_m)./(Nak_m.^(Nak_m-0.5)*rou_rms.^(2*Nak_m-1).*fd*sqrt(2*pi));  %% 理论AFD曲线, 持续时间(s)/每个电平
    semilogy(dB_range, levelt, '*', dB_range, AFD,'r:');
    xlabel('\rho_r_m_s(dB)'),ylabel('AFD(s)');grid on;
elseif  Plot_flag == 1
%     figure;
    semilogy(dB_range,levelt,'-*');
    xlabel('\rho_r_m_s(dB)'),ylabel('AFD(s)');grid on;
end

