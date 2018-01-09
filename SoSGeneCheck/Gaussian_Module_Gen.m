function H = Gaussian_Module_Gen(fs,fmax,endT,N,mean,var_2,delta_omega, plot_flag)
% %----生成一个均值为mean方差为var_2的高斯变量----%
% close all; clear all; clc;
% fs = 1000000;  % 采样率
% fmax = 50;  % 最大多普勒频率
% fd = 100;   % 中心频偏
% endT = 1;  % 仿真时间
% N = 32;    % 支路数
% mean = 0;
% var_2 = 1;% 功率
% delta_omega = 0; % 多普勒频率随机偏移量
% plot_flag = 1;

%% 产生一个服从N（0,1）分布的高斯变量
alpha_n = linspace(0, 2*pi-2*pi/(N+1), N);         % 入射角均匀分布
omega_n_I = 2*pi*fmax*cos(alpha_n) + delta_omega;  % 偏移量非常微小
phi_n = unifrnd(0,2*pi,1,2*N);
phi_n_I = phi_n(1:2:2*N);
t = 1/fs:1/fs:endT;
cos_value = cos(omega_n_I'*t + phi_n_I'*ones(1,fs*endT));
H_norm = sqrt(2/N) * sum(cos_value);
%% 产生一个服从N（mean,var_2）分布的高斯变量
H = sqrt(var_2)*H_norm + mean;

if plot_flag==1
    figure;
    plot(t, H);xlim([0 1]);
    xlabel('t');ylabel('H(t)');
    
    s = H; % 高斯分布
    delta_x = 0.1;
    x = -5:delta_x:5;
    pdf_ideal = normpdf(x,mean,sqrt(var_2));
    figure;
    plot(x, pdf_ideal);
    hold on;
    
    delta_x = 0.2;
    x = -5:delta_x:5;
    pdf_stat = hist(s,x);
    pdf_sim = pdf_stat/(length(s)*delta_x);
    plot(x,pdf_sim, '*');
    xlabel('x');ylabel('Gaussian PDF');legend('Theoretical','Simulated');
end