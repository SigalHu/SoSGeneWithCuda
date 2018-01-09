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

var_2 = Nak_omega/2/Nak_m;

fid = fopen("C:\Code\Team\SoSGeneWithCuda\SoSGeneWithCuda\SoSGeneWithCuda\nakagamiIQ.bin",'rb');
H = fread(fid,inf,'float').';
fclose(fid);
H_complex =  H(1:end/2) + 1i*H(end/2+1:end);
H = abs(H_complex);

Nak_pdf =  @(rr,pp,mm) (2/gamma(mm)) * (mm/pp)^mm .* rr.^(2*mm-1).* exp(-mm*rr.^2/pp);
if plot_flag == 1
    figure;
    t = 1/fs : 1/fs : endT;
    plot(t, H);xlim([0 1]);
    xlabel('t');ylabel('H(t)');
    delta_x = 0.1;
    x = 0:delta_x:10;
    pdf_sim =  hist(H,x)/(length(H) * delta_x);
    figure;
    plot(x, pdf_sim, 'r*');
    hold on;
    
    delta_x = delta_x / 10;
    x = 0:delta_x:10;
    pdf_ideal = Nak_pdf(x, Nak_omega, Nak_m);
    plot(x, pdf_ideal);
    xlabel('x');ylabel('Nakagami PDF');
    legend('Simulated','Theoretical');
    
    %%--PSD--%%
    num = 500;
    [Pxx4,fpoint] = pmtm(H_complex,3.5,1000,fs);
    Pxx5 = fftshift(Pxx4);
    fpoint1 = fpoint-num;
    
    figure;plot(fpoint1, 10*log10(Pxx5/max(Pxx5)));hold on;
    Pxx_theory = var_2./sqrt(1-(fpoint1(num-fmax:num+fmax+1)/fmax).^2)./(pi*fmax);
    Pxx_theory(find(Pxx_theory == Inf)) = var_2./sqrt(0.001)./(pi*fmax);
    plot(fpoint1(num-fmax:num+fmax+1), 10*log10(Pxx_theory/max(Pxx_theory)),'r-*');xlim([-300 300]);
    xlabel('f(Hz)');ylabel('PSD(dB)');legend('MESD','Theory');
    
    dB_range = [-10:0.5:5];
    Plot_LCR_time(H,fs,dB_range,fd,Nak_m,2);
    Plot_AFD_time(H,fs,dB_range,fd,Nak_m,2);
end
