close all; clear all; clc;
fs = 1000;  % 采样率
fmax = 50;  % 最大多普勒频率
fd = 100;   % 中心频偏
endT = 100;  % 仿真时间
N = 32;    % 支路数
Nak_m = 1;
Nak_omega = 1;
delta_omega = 0; % 多普勒频率随机偏移量
plot_flag = 1;

fid = fopen("C:\Code\Team\SoSGeneWithCuda\SoSGeneWithCuda\SoSGeneWithCuda\nakagami.bin",'rb');
H = fread(fid,inf,'float');
fclose(fid);

Nak_pdf =  @(rr,pp,mm) (2/gamma(mm)) * (mm/pp)^mm .* rr.^(2*mm-1).* exp(-mm*rr.^2/pp);
if plot_flag == 1
    figure;
    t = 1/fs : 1/fs : endT;
    plot(t, H);xlim([0 1]);
    xlabel('t');ylabel('H(t)');
    s = abs(H);
    delta_x = 0.1;
    x = 0:delta_x:10;
    pdf_sim =  hist(s,x)/(length(s) * delta_x);
    figure;
    plot(x, pdf_sim, 'r*');
    hold on;
    
    delta_x = delta_x / 10;
    x = 0:delta_x:10;
    pdf_ideal = Nak_pdf(x, Nak_omega, Nak_m);
    plot(x, pdf_ideal);
    xlabel('x');ylabel('Nakagami PDF');
    legend('Simulated','Theoretical');
end
