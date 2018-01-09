%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LCR--电平通过率，信道包络每秒通过并低于某电平的平均次数
% H  输入信道矩阵，行数-信道数目，列数-信道持续点数 ；
% fs 采样率
% dB_range 统计参考电平的范围,相对包络均方根的比值；
% fd 最大多普勒频率，求瑞利信道理论值时需要；
% Plot_flag '2'-画图,并与瑞利信道理论值比较 ‘1’-只画图  '0'-不画图
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sumlevel level_val] = Plot_LCR_time(H,fs,dB_range,fd,Nak_m,Plot_flag)

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
sumlevel = zeros(line_n,length(rou_rms));
if nargout == 2
    level_val = zeros(line_n,length(rou_rms));
end
for method = 1:line_n
    nn = 0;
    for level = sqrt( (rou_rms.^2)*mean(abs(H(method,:)).^2) )  %% 比较电平，相对包络均方根的值
        nn = nn+1;
        for n = 1:row_n-1
            if abs(H(method,n)) > level && abs(H(method,n+1)) <= level
                sumlevel(method,nn) = sumlevel(method,nn)+1;
            elseif abs(H(method,n)) < level && abs(H(method,n+1)) >= level
                sumlevel(method,nn) = sumlevel(method,nn)+1;
            end
        end
    end
    sumlevel(method,:) = sumlevel(method,:)./(row_n/fs);
    if nargout == 2
        level_val(method,:) = sqrt( (rou_rms.^2)*mean(abs(H(method,:)).^2) ); 
    end
end

if Plot_flag == 2
%     LCR = sqrt(2*pi)*fd*rou_rms.*exp(-rou_rms.^2);
    LCR = sqrt(2*pi)*fd*Nak_m.^(Nak_m-0.5)*rou_rms.^(2*Nak_m-1).*exp(-Nak_m*rou_rms.^2)/gamma(Nak_m);
    figure;
    semilogy(dB_range,sumlevel,'*-',dB_range,LCR,'r:');
    xlabel('\rho_r_m_s(dB)'),ylabel('LCR(次/s)');grid on;
elseif  Plot_flag == 1
%     figure;
    semilogy(dB_range,sumlevel,'-*');
    xlabel('\rho_r_m_s(dB)'),ylabel('LCR(次/s)');grid on;
end
