function [Nak_Lognorm_H] = Gen_shadow_fade_var_func(fs,endT,fd_fast,fd_slow,Shadow_dB,Pavg,Nak_m,mode,Gamfadmode)

if nargin == 7
    mode = 1;
    Gamfadmode = 12;
elseif nargin == 8
    Gamfadmode = 12;
end

% clear all;  clc; close all;
% fs = 10000;       endT = 100;
% fd_fast = 20;   fd_slow = 10;
% Shadow_dB = 4.2;  Pavg = 1; 
% Nak_m = 9.6; 
% mode = 2;
% Gamfadmode = 12;

Nak_H = Gen_AutoCorNak_SimpFunc(fs,fd_fast,endT,Nak_m,1); % abs(wgn(1,fs*endT,1, 'linear','complex'));% 
switch mode
    case 1 % Nak-Lognormal fading(Modified Suzuki) 
        sigX = Shadow_dB/8.686; miuX = log(Pavg);   
        Lognorm_H = Single_Rician_SoS(fs,fd_slow,0,0,endT); % wgn(1,fs*endT,0.5, 'linear');% 
        Lognorm_H = exp(real(Lognorm_H)*sqrt(2)*sigX + miuX);
        Nak_Lognorm_H = Nak_H .* sqrt(Lognorm_H);
    case 2 % Nak-Lognormal --> Nak-Gamma fading
        sigX = Shadow_dB/8.686; miuX = log(Pavg); 
        Gam_ms = 1/(exp(sigX^2)-1); 
        Gam_ps = Pavg*sqrt((1+Gam_ms)/Gam_ms);
        Lognorm_H = Gen_AutoCorNak_SimpFunc(fs,fd_slow,endT,Gam_ms,Gam_ps); 
        Lognorm_H = Lognorm_H.^2; % gamrnd(Gam_ms,Gam_ps/Gam_ms,1,endT*fs);%  
        Nak_Lognorm_H = Nak_H .* sqrt(Lognorm_H);
    case 3 % Nak-Lognormal Nak*sqrt(lognorm)--> Nak-Gamma fading Nak*Nak--> sqrt of Gamma fading Nak
        sigX = Shadow_dB/8.686; miuX = log(Pavg); 
        Gam_ms = 1/(exp(sigX^2)-1); 
        Gam_ps = Pavg*sqrt((1+Gam_ms)/Gam_ms);
        K1 = (Nak_m+1)*(Gam_ms+1)/(Nak_m*Gam_ms); K2 = (Nak_m+2)*(Gam_ms+2)/(Nak_m*Gam_ms);
        if( Gamfadmode == 12 )
            Gam_m = 1/(K1-1); Gam_sig = Gam_ps;
        elseif( Gamfadmode == 13 )
            Gam_m = 4/(-3+sqrt(9+8*(K1*K2-1))); Gam_sig = Gam_ps;
        end
        Nak_Lognorm_H = Gen_AutoCorNak_SimpFunc(fs,fd_slow,endT,Gam_m,Gam_sig); 
        %Nak_Lognorm_H = sqrt(gamrnd(Gam_m,Gam_sig/Gam_m,1,endT*fs));  
    otherwise
end



Nak_pdf =  @(rr,pp,mm) (2/gamma(mm)) * (mm/pp)^mm .* rr.^(2*mm-1) .* exp(-mm*rr.^2/pp);
Nak_pdf_p = @(rr,pp,mm) (2/gamma(mm)) * (mm./pp).^mm * rr^(2*mm-1) .* exp(-mm*rr^2./pp);
Gama_pdf =  @(rr,pp,mm) (rr.^(mm-1) .* (1/gamma(mm)) * (mm/pp)^mm .* exp(-mm*rr/pp));
Logn_pdf_sig = @(rr,sigsig,miumiu) 1/(sqrt(2*pi)*sigsig*rr) .* exp(-(log(rr)-miumiu)^2/(sqrt(2)*sigsig).^2);
GK_pdf = @(mm,msms,sigssigs,rr) 4/(gamma(mm)*gamma(msms)) * (mm*msms/sigssigs)^((mm+msms)/2) * ...
                               ( rr.^(mm+msms-1) .* besselk(msms-mm,2*rr*sqrt(mm*msms/sigssigs)) );
% %%--Confirm Nakagami and Lognormal PDF--%%
% Gam_ms = 1/(exp(sigX^2)-1); 
% Gam_ps = Pavg*sqrt((1+Gam_ms)/Gam_ms);
% r = [0.001:0.05:max(Lognorm_H)];
% figure
% pdf_stat = hist(Lognorm_H,r);
% subplot(2,1,1),plot(r,pdf_stat/(length(Lognorm_H)*(r(2)-r(1))),'*'); hold on;
% subplot(2,1,1),plot(r,lognpdf(r,miuX,sigX)); 
% subplot(2,1,1),plot(r,gampdf(r,Gam_ms,Gam_ps/Gam_ms),'r'); 
% legend('Sim','Lognorm','Gama');
% r = [0.001:0.05:max(Nak_H)];
% pdf_stat = hist(Nak_H,r);
% subplot(2,1,2),plot(r,pdf_stat/(length(Nak_H)*(r(2)-r(1))),'*'); hold on;
% subplot(2,1,2),plot(r,Nak_pdf(r,1,Nak_m)); 

% %--Confirm Nakagami-Lognormal PDF--%%  
% Gam_ms = 1/(exp(sigX^2)-1); 
% Gam_ps = Pavg*sqrt((1+Gam_ms)/Gam_ms);
% r = 0:max(Nak_Lognorm_H)/100:max(Nak_Lognorm_H);
% for m = 1:length(r)
%     start_afi = 0.0001; deta_afi = 0.005; end_afi = max(Nak_Lognorm_H);
%     tmp = Nak_pdf_p(r(m),[start_afi:deta_afi:end_afi],Nak_m).*lognpdf([start_afi:deta_afi:end_afi],miuX,sigX);
%     NakLogn_pdf(m) = (sum(tmp)-tmp(1)-tmp(end))*deta_afi;
% end
% figure;
% pdf_stat = hist(Nak_Lognorm_H,r);
% plot(r,pdf_stat/(length(Nak_Lognorm_H)*(r(2)-r(1))),'*'); 
% hold on;
% plot(r,NakLogn_pdf);
% hold on;
% plot(r,GK_pdf(Nak_m,Gam_ms,Gam_ps,r),'.');
% legend('Sim','Nak-Longnorm','Nak-Gamma(G-K)');



