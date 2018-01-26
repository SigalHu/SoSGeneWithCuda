clear all;  close all; clc;

run_time = 100;

Nak_m = [11.3 10.3 9.2 9.6];
Shadow_dB = [1.1 3.2 3.9 4.2];

fs = 10000;
endT = 100;
fd_fast = 50;
fd_slow = 50;

Pavg_Chan = 1;

Nak_Lognorm_H = zeros(1,fs*endT,'single');
for ii=1:run_time
    for jj = 1:length(Nak_m)
        sigX = Shadow_dB(jj)/8.686;
        miuX = log(Pavg_Chan);
        Gam_ms = 1/(exp(sigX^2)-1);
        Gam_ps = Pavg_Chan*sqrt((1+Gam_ms)/Gam_ms);
        
        Nak_Lognorm_H = Gen_shadow_fade_var_func(fs,endT,fd_fast,fd_slow,Shadow_dB(jj),Pavg_Chan,Nak_m(jj),2);
        filename = ['lognak',num2str(jj-1),'_',num2str(ii),'.bin'];
        fid = fopen(filename,'wb');
        fwrite(fid,Nak_Lognorm_H,'single');
        fclose(fid);     
    end
end