close all; clear all; clc;
fs = 1000;  % ������
fmax = 50;  % ��������Ƶ��
fd = 100;   % ����Ƶƫ
endT = 1000;  % ����ʱ��
N = 32;    % ֧·��
mean = 0;
var_2 = 1;% ����
delta_omega = 0; % ������Ƶ�����ƫ����
plot_flag = 1;

fid = fopen("C:\Code\Team\SoSGeneWithCuda\SoSGeneWithCuda\SoSGeneWithCuda\lognormal.bin",'rb');
H = fread(fid,inf,'float');
fclose(fid);

if plot_flag==1
    figure;
    t = 1/fs : 1/fs : endT;
    plot(t, abs(H));xlim([0 1]);
    xlabel('t');ylabel('H(t)');
    s = H; % ��˹�ֲ�
    delta_x = 0.2;
    x = 0:delta_x:10;
    pdf_sim = hist(s,x)/(length(s)*delta_x);
    figure;
    plot(x, pdf_sim, 'r*');
    hold on;
    
    delta_x = 0.01;
    x = 0:delta_x:10;
    pdf_ideal = exp(-(log(x) - mean).^2 / (2 * var_2))./ (sqrt(2*pi)* sqrt(var_2) .* x);
    plot(x, pdf_ideal);
    xlabel('x');ylabel('LogNormal PDF');
    legend('Theoretical','Simulated');
end