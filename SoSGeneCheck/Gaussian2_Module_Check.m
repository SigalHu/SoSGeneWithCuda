close all; clear all; clc;
plot_flag = 1;
fs = 1000; 
fd = 50; 
endT = 1000;
N = 32;       % ֧·��
PSD = 0;
delta_fd = 0;
delta_omega = 0;

fid = fopen("C:\Code\Team\SoSGeneWithCuda\SoSGeneWithCuda\x64\Release\gaussian2.bin",'rb');
H = fread(fid,inf,'float');
fclose(fid);
Hc = H(1:end/2);
Hs = H(end/2+1:end);
H =  (Hc + 1i*Hs);

if plot_flag == 1
    %%--ģ��ͳ��������֤��matlab�����и�˹����������Ķ���һ��
    %--PDF--%
    s = Hc; % ��˹�ֲ�
    x = -5:0.1:5;
    pdf_ideal = normpdf(x,0,sqrt(1));    %% ��˹���۸��ʷֲ�
    pdf_stat = hist(s,x);                  %% ʵ��ͳ�Ƹ��ʷֲ� 
    subplot(2,1,1),plot(x,pdf_ideal,x,pdf_stat/(length(s)*0.1),'*');%%����pdf�µ����Ϊ1��������ͳ�Ƶ�pdf��ֵ֮��Ϊ1���������������й�
    title('PDF(Gauss)'),legend('Theroy','Sim');
    s = abs(H); % �����ֲ�
    x = 0:0.1:5;
    pdf_ideal = raylpdf(x,sqrt(1));         %% ��һ���������۸��ʷֲ�
    pdf_stat = hist(s,x);           
    subplot(2,1,2),plot(x,pdf_ideal,x,pdf_stat/(length(s)*0.1),'*');
    title('PDF(Rayleigh)'),legend('Theroy','Sim');
    %--ACF--%
    maxlags = 10*fs/fd;
    [acf(1:2*maxlags+1) lags] = xcov(Hc,maxlags,'coeff');
    acf_ref = besselj(0,2*pi*fd*[-maxlags/fs:1/fs:maxlags/fs]) ;
    figure
    subplot(2,1,1),plot(lags/(fs/fd),acf_ref,lags/(fs/fd),acf,'*');
    title('ACF');
    legend('Theory','Sim');
    %--CCF--%
    [ccf(1:2*maxlags+1) lags] = xcov(Hc,Hs,maxlags,'coeff');
    subplot(2,1,2),plot(lags,zeros(1,length(lags)),lags,ccf,'*');
    legend('Theory','Sim');
    title('CCF');
end