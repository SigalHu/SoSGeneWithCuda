%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LCR--��ƽͨ���ʣ��ŵ�����ÿ��ͨ��������ĳ��ƽ��ƽ������
% H  �����ŵ���������-�ŵ���Ŀ������-�ŵ��������� ��
% fs ������
% dB_range ͳ�Ʋο���ƽ�ķ�Χ,��԰���������ı�ֵ��
% fd ��������Ƶ�ʣ��������ŵ�����ֵʱ��Ҫ��
% Plot_flag '2'-��ͼ,���������ŵ�����ֵ�Ƚ� ��1��-ֻ��ͼ  '0'-����ͼ
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

rou_rms = 10.^(dB_range/20);  %%���Ա�ֵ rou_rms^2 = level^2 / Power
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
    for level = sqrt( (rou_rms.^2)*mean(abs(H(method,:)).^2) )  %% �Ƚϵ�ƽ����԰����������ֵ
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
    xlabel('\rho_r_m_s(dB)'),ylabel('LCR(��/s)');grid on;
elseif  Plot_flag == 1
%     figure;
    semilogy(dB_range,sumlevel,'-*');
    xlabel('\rho_r_m_s(dB)'),ylabel('LCR(��/s)');grid on;
end
