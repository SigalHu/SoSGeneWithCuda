close all;clear all;clc;
plot_flag = 1;
fs = 1000;
fmax = 48.4; 
endT = 100;    % 产生一个大周期的数据
N_Path = 32; % 散射支路数
Nak_m = 10.3;
Nak_omega = 1;
Shadow_dB = 3.2;  
Pavg = 1;

sigX = Shadow_dB/8.686; 
miuX = log(Pavg);
var_2 = Nak_omega/2/Nak_m;

fid = fopen("C:\Code\Team\SoSGeneWithCuda\SoSGeneWithCuda\x64\Release\lognakIQ1_2.bin",'rb');
H = fread(fid,inf,'float').';
fclose(fid);
H_complex = H(1:end/2) + 1i*H(end/2+1:end);
H = abs(H_complex);
    
Nak_pdf =  @(rr,pp,mm) (2/gamma(mm)) * (mm/pp)^mm .* rr.^(2*mm-1) .* exp(-mm*rr.^2/pp);
Nak_pdf_p = @(rr,pp,mm) (2/gamma(mm)) * (mm./pp).^mm * rr^(2*mm-1) .* exp(-mm*rr^2./pp);
Gama_pdf =  @(rr,pp,mm) (rr.^(mm-1) .* (1/gamma(mm)) * (mm/pp)^mm .* exp(-mm*rr/pp));
Logn_pdf_sig = @(rr,sigsig,miumiu) 1/(sqrt(2*pi)*sigsig*rr) .* exp(-(log(rr)-miumiu)^2/(sqrt(2)*sigsig).^2);
GK_pdf = @(mm,msms,sigssigs,rr) 4/(gamma(mm)*gamma(msms)) * (mm*msms/sigssigs)^((mm+msms)/2) * ...
                               ( rr.^(mm+msms-1) .* besselk(msms-mm,2*rr*sqrt(mm*msms/sigssigs)) );

%--Confirm Nakagami-Lognormal PDF--%%  
Gam_ms = 1/(exp(sigX^2)-1); 
Gam_ps = Pavg*sqrt((1+Gam_ms)/Gam_ms);
r = 0:max(H)/100:max(H);
for m = 1:length(r)
    start_afi = 0.0001; deta_afi = 0.005; end_afi = max(H);
    tmp = Nak_pdf_p(r(m),[start_afi:deta_afi:end_afi],Nak_m).*lognpdf([start_afi:deta_afi:end_afi],miuX,sigX);
    NakLogn_pdf(m) = (sum(tmp)-tmp(1)-tmp(end))*deta_afi;
end
figure
pdf_stat = hist(H,r);
plot(r,pdf_stat/(length(H)*(r(2)-r(1))),'*'); 
hold on;
plot(r,NakLogn_pdf);
% hold on;
% plot(r,GK_pdf(Nak_m,Gam_ms,Gam_ps,r),'.');
legend('Sim','Nak-Longnorm');

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
Plot_LCR_time(H,fs,dB_range,fmax,Nak_m,2);
Plot_AFD_time(H,fs,dB_range,fmax,Nak_m,2);





