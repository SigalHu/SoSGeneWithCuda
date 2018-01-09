function H = Gaussian_Module_Gen(fs,fmax,endT,N,mean,var_2,delta_omega, plot_flag)
% %----����һ����ֵΪmean����Ϊvar_2�ĸ�˹����----%
% close all; clear all; clc;
% fs = 1000000;  % ������
% fmax = 50;  % ��������Ƶ��
% fd = 100;   % ����Ƶƫ
% endT = 1;  % ����ʱ��
% N = 32;    % ֧·��
% mean = 0;
% var_2 = 1;% ����
% delta_omega = 0; % ������Ƶ�����ƫ����
% plot_flag = 1;

%% ����һ������N��0,1���ֲ��ĸ�˹����
alpha_n = linspace(0, 2*pi-2*pi/(N+1), N);         % ����Ǿ��ȷֲ�
omega_n_I = 2*pi*fmax*cos(alpha_n) + delta_omega;  % ƫ�����ǳ�΢С
phi_n = unifrnd(0,2*pi,1,2*N);
phi_n_I = phi_n(1:2:2*N);
t = 1/fs:1/fs:endT;
cos_value = cos(omega_n_I'*t + phi_n_I'*ones(1,fs*endT));
H_norm = sqrt(2/N) * sum(cos_value);
%% ����һ������N��mean,var_2���ֲ��ĸ�˹����
H = sqrt(var_2)*H_norm + mean;

if plot_flag==1
    figure;
    plot(t, H);xlim([0 1]);
    xlabel('t');ylabel('H(t)');
    
    s = H; % ��˹�ֲ�
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