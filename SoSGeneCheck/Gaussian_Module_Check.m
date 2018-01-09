close all; clear all; clc;
fs = 1000;  % 采样率
fmax = 50;  % 最大多普勒频率
fd = 100;   % 中心频偏
endT = 1000;  % 仿真时间
N = 32;    % 支路数
mean = 0;
var_2 = 1;% 功率
delta_omega = 0; % 多普勒频率随机偏移量
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
    s = H; % 高斯分布
    delta_x = 0.1;
    x = -5:delta_x:5;
    pdf_ideal = normpdf(x,mean,sqrt(var_2));
    pdf_stat = hist(s,x);
    pdf_sim = pdf_stat/(length(s)*delta_x);
    subplot(212);
    plot(x, pdf_ideal, x, pdf_sim, '*');
    xlabel('x');ylabel('Gaussian PDF');legend('Theoretical','Simulated');
end