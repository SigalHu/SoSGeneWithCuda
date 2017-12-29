close all;clear all;clc;
plot_flag = 1;
fs = 1000;
fd = 50; 
endT = 1000;    % 产生一个大周期的数据
N_Path = 32; % 散射支路数
Nak_Power = 1;
Nak_m = 10.3;
Shadow_dB = 3.2;  
Pavg = 1;

sigX = Shadow_dB/8.686; 
miuX = log(Pavg);

fid = fopen("C:\Code\Team\SoSGeneWithCuda\SoSGeneWithCuda\x64\Release\lognak.bin",'rb');
Nak_Lognorm_H = fread(fid,inf,'float');
fclose(fid);
    
Nak_pdf =  @(rr,pp,mm) (2/gamma(mm)) * (mm/pp)^mm .* rr.^(2*mm-1) .* exp(-mm*rr.^2/pp);
Nak_pdf_p = @(rr,pp,mm) (2/gamma(mm)) * (mm./pp).^mm * rr^(2*mm-1) .* exp(-mm*rr^2./pp);
Gama_pdf =  @(rr,pp,mm) (rr.^(mm-1) .* (1/gamma(mm)) * (mm/pp)^mm .* exp(-mm*rr/pp));
Logn_pdf_sig = @(rr,sigsig,miumiu) 1/(sqrt(2*pi)*sigsig*rr) .* exp(-(log(rr)-miumiu)^2/(sqrt(2)*sigsig).^2);
GK_pdf = @(mm,msms,sigssigs,rr) 4/(gamma(mm)*gamma(msms)) * (mm*msms/sigssigs)^((mm+msms)/2) * ...
                               ( rr.^(mm+msms-1) .* besselk(msms-mm,2*rr*sqrt(mm*msms/sigssigs)) );

%--Confirm Nakagami-Lognormal PDF--%%  
Gam_ms = 1/(exp(sigX^2)-1); 
Gam_ps = Pavg*sqrt((1+Gam_ms)/Gam_ms);
r = [0.001:0.02:max(Nak_Lognorm_H)];
for m = 1:length(r)
    start_afi = 0.0001; deta_afi = 0.005; end_afi = max(Nak_Lognorm_H);
    tmp = Nak_pdf_p(r(m),[start_afi:deta_afi:end_afi],Nak_m).*lognpdf([start_afi:deta_afi:end_afi],miuX,sigX);
    NakLogn_pdf(m) = (sum(tmp)-tmp(1)-tmp(end))*deta_afi;
end
figure
pdf_stat = hist(Nak_Lognorm_H,r);
plot(r,pdf_stat/(length(Nak_Lognorm_H)*(r(2)-r(1))),'*'); hold on;
plot(r,NakLogn_pdf);hold on;
plot(r,GK_pdf(Nak_m,Gam_ms,Gam_ps,r),'.');
legend('Sim','Nak-Longnorm','Nak-Gamma(G-K)');





