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

fid = fopen("C:\Code\Team\SoSGeneWithCuda\SoSGeneWithCuda\SoSGeneWithCuda\gaussian.bin",'rb');
H = fread(fid,inf,'float');
fclose(fid);

if plot_flag==1
    figure;
    subplot(211);
    t = 1/fs:1/fs:endT;
    plot(t, H);xlim([0 1]);
    xlabel('t');ylabel('H(t)');
    s = H; % ��˹�ֲ�
    delta_x = 0.1;
    x = -5:delta_x:5;
    pdf_ideal = normpdf(x,mean,sqrt(var_2));
    pdf_stat = hist(s,x);
    pdf_sim = pdf_stat/(length(s)*delta_x);
    subplot(212);
    plot(x, pdf_ideal, x, pdf_sim, '*');
    xlabel('x');ylabel('Gaussian PDF');legend('Theoretical','Simulated');
end