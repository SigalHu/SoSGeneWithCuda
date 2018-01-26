clear all;  close all; clc;

n_times = 1000;
modulation = 'QPSK'; % BPSK QPSK 8PSK 16QAM 64QAM
snr_dB_min = 3;
hop_N = 2;
d_A2A = 2e3;
d_G2A = 500;
d_A2G = 500;
h = 200:40:1e3;

ParaList = [11.3 1.1 10.3 3.2; ...    % hu 丘陵
            11.3 1.1 9.2  3.9; ...    % hu 山区
            11.3 1.1 9.6  4.2];       % hu 海洋
% ParaList = [1 3 1 3; ...    % hu 丘陵
%             1 3 1 3; ...    % hu 山区
%             1 3 1 3];       % hu 海洋

SimulationColorList = {'r.:','b.:','k.:','g.:';
                       'r*:','b*:','k*:','g*:';
                       'ro:','bo:','ko:','go:';};
TheoryColorList = {'r-','b-','k-','g-'};

fc = 968e6;
fs = 1000;
endT = 100;
fd = 48.4;

LaunPower_dB_G1 = 0;
RepeaterGain_dB_An = 0;

AnteGain_dB_G1 = 0;
AnteGain_dB_An = 0;
AnteGain_dB_G2 = 0;

Sum_Hu = @(a,sumLoc) arrayfun(@(loc) sum(a(1:loc)),sumLoc);
% hu 理论表达式
PathLoss_Calc = @(d,f) 32.44 + 20*log10(d/1e3) + 20*log10(f/1e6);
Nak_pdf_p = @(rr,pp,mm) (2/gamma(mm)) * (mm./pp).^mm * rr^(2*mm-1) .* exp(-mm*rr^2./pp);
GK_pdf = @(rr,mm,msms,sigssigs) 4/gamma(mm)/gamma(msms)*(mm*msms/sigssigs)^((mm+msms)/2)*rr.^(mm+msms-1).*besselk(msms-mm,2*sqrt(mm*msms/sigssigs)*rr);
CascadedGG_pdf = @(xx,mm1,mm2,msms1,msms2,Ravg) arrayfun(@(xx) double(feval(symengine,'meijerG','[[], []]', ['[[',num2str(msms1),',',num2str(mm1),',',num2str(msms2),',',num2str(mm2),'], []]'], num2str(xx))), mm1*msms1*mm2*msms2/Ravg*xx.^2)./(gamma(mm1)*gamma(msms1)*gamma(mm2)*gamma(msms2)*xx/2);
CascadedGG_cdf = @(rr,mm1,mm2,msms1,msms2,Ravg) arrayfun(@(rr) double(feval(symengine,'meijerG','[[1], []]', ['[[',num2str(msms1),',',num2str(mm1),',',num2str(msms2),',',num2str(mm2),'], [0]]'], num2str(rr))), mm1*msms1*mm2*msms2/Ravg*rr)./(gamma(mm1)*gamma(msms1)*gamma(mm2)*gamma(msms2));
CascadedGG_ser = @(mm1,mm2,msms1,msms2,Ravg,MPSK) arrayfun(@(rr) double(feval(symengine,'meijerG','[[1,1/2], []]', ['[[',num2str(msms1),',',num2str(mm1),',',num2str(msms2),',',num2str(mm2),'], [0]]'], num2str(rr))), mm1*msms1*mm2*msms2/sin(pi/MPSK)^2./Ravg)/(gamma(mm1)*gamma(msms1)*gamma(mm2)*gamma(msms2)*sqrt(pi));
CascadedGG_ser_BPSK = @(mm1,mm2,msms1,msms2,Ravg) arrayfun(@(rr) double(feval(symengine,'meijerG','[[1,1/2], []]', ['[[',num2str(msms1),',',num2str(mm1),',',num2str(msms2),',',num2str(mm2),'], [0]]'], num2str(rr))), mm1*msms1*mm2*msms2./Ravg)/(2*gamma(mm1)*gamma(msms1)*gamma(mm2)*gamma(msms2)*sqrt(pi));
% CascadedGG_C = @(mm1,mm2,msms1,msms2,Ravg) arrayfun(@(rr) double(feval(symengine,'meijerG','[[0], [1]]', ['[[',num2str(msms1),',',num2str(mm1),',',num2str(msms2),',',num2str(mm2),'], [0,0]]'], num2str(rr))), mm1*msms1*mm2*msms2./Ravg)/(log(2)*gamma(mm1)*gamma(msms1)*gamma(mm2)*gamma(msms2));
CascadedGG_C = @(mm1,mm2,msms1,msms2,Ravg) arrayfun(@(rr) double(feval(symengine,'meijerG','[[0], [1]]', ['[[',num2str(msms1),',',num2str(mm1),',',num2str(msms2),',',num2str(mm2),',0,0], []]'], num2str(rr))), mm1*msms1*mm2*msms2./Ravg)/(log(2)*gamma(mm1)*gamma(msms1)*gamma(mm2)*gamma(msms2));

% hu G2A参数
PathLen_G2A = sqrt(d_G2A^2+h.^2);
Pavg_Chan_G2A = 1;

% hu A2A参数
PathLen_A2A = d_A2A;

% hu A2G参数
PathLen_A2G = sqrt(d_A2G^2+h.^2);
Pavg_Chan_A2G = 1;

ser_sim = zeros(size(ParaList,1),length(h));
ser_theory = zeros(size(ParaList,1),length(h));

for ii=1:n_times
    %% hu 发射机
    LaunPower_G1 = 10^(LaunPower_dB_G1/10);
    switch modulation
        case 'BPSK'
            DataSorce = randi([0,1],1,fs*endT,'single');
            LaunSignal = pskmod(DataSorce,2);
        case 'QPSK'
            DataSorce = randi([0,3],1,fs*endT,'single');
            LaunSignal = pskmod(DataSorce,4);
        case '8PSK'
            DataSorce = randi([0,7],1,fs*endT,'single');
            LaunSignal = pskmod(DataSorce,8);
        case '16QAM'
            DataSorce = randi([0,15],1,fs*endT,'single');
            LaunSignal = qammod(DataSorce,16);
        case '64QAM'
            DataSorce = randi([0,63],1,fs*endT,'single');
            LaunSignal = qammod(DataSorce,64);
    end
    LaunSignal = LaunSignal * sqrt(LaunPower_G1/(mean(abs(LaunSignal).^2)));
    
    %% hu 信道
    % hu 确定值部分
    % hu G2A
    AnteGainLaun_G1 = AnteGain_dB_G1;
    AnteGainRecv_A1 = AnteGain_dB_An;
    
    PathLoss_dB_G2A = PathLoss_Calc(PathLen_G2A,fc);
    RecvPower_dB_A1 = LaunPower_dB_G1 + AnteGainLaun_G1 + AnteGainRecv_A1 - PathLoss_dB_G2A;
    
    % hu A2A
    for nn = 1:hop_N-1
        % hu A2A
        AnteGainLaun_A1 = AnteGain_dB_An;
        AnteGainRecv_A2 = AnteGain_dB_An;
        
        LaunPower_dB_A1 = RecvPower_dB_A1 + RepeaterGain_dB_An;
        PathLoss_dB_A2A = PathLoss_Calc(PathLen_A2A,fc);
        RecvPower_dB_A1 = LaunPower_dB_A1 + AnteGainLaun_A1 + AnteGainRecv_A2 - PathLoss_dB_A2A;
    end
    RecvPower_dB_A2 = RecvPower_dB_A1;
    
    % hu A2G
    AnteGainLaun_A2 = AnteGain_dB_An;
    AnteGainRecv_G2 = AnteGain_dB_G2;
    
    LaunPower_dB_A2 = RecvPower_dB_A2 + RepeaterGain_dB_An;
    PathLoss_dB_A2G = PathLoss_Calc(PathLen_A2G,fc);
    RecvPower_dB_G2 = LaunPower_dB_A2 + AnteGainLaun_A2 + AnteGainRecv_G2 - PathLoss_dB_A2G;
    RecvPower_G2 = 10.^(RecvPower_dB_G2/10);
    
    % hu 随机值部分
    % hu 产生噪声
    Shadow_dB_G2A = min(ParaList(:,2));
    sigX_G2A = Shadow_dB_G2A/8.686;
    miuX_G2A = log(Pavg_Chan_G2A);
    Gam_ms_G2A = 1/(exp(sigX_G2A^2)-1);
    Gam_ps_G2A = Pavg_Chan_G2A*sqrt((1+Gam_ms_G2A)/Gam_ms_G2A);
    
    Shadow_dB_A2G = min(ParaList(:,4));
    sigX_A2G = Shadow_dB_A2G/8.686;
    miuX_A2G = log(Pavg_Chan_A2G);
    Gam_ms_A2G = 1/(exp(sigX_A2G^2)-1);
    Gam_ps_A2G = Pavg_Chan_A2G*sqrt((1+Gam_ms_A2G)/Gam_ms_A2G);
    
    snr_min = 10^(snr_dB_min/10);
    N0 = min(RecvPower_G2 * Gam_ps_G2A * Gam_ps_A2G)/snr_min;
    
    for mm = 1:size(ParaList,1)
        Nak_m_G2A = ParaList(mm,1);
        Shadow_dB_G2A = ParaList(mm,2);
        Nak_m_A2G = ParaList(mm,3);
        Shadow_dB_A2G = ParaList(mm,4);
        
        % hu G2A
        sigX_G2A = Shadow_dB_G2A/8.686;
        miuX_G2A = log(Pavg_Chan_G2A);
        Gam_ms_G2A = 1/(exp(sigX_G2A^2)-1);
        Gam_ps_G2A = Pavg_Chan_G2A*sqrt((1+Gam_ms_G2A)/Gam_ms_G2A);
        
        
        fid = fopen(['C:\Code\Team\SoSGeneWithCuda\SoSGeneWithCuda\x64\Release\lognak0_',num2str(ii),'.bin'],'rb');
        H = fread(fid,inf,'float').';
        fclose(fid);
        Nak_Lognorm_H = H;
        
        % hu A2G
        sigX_A2G = Shadow_dB_A2G/8.686;
        miuX_A2G = log(Pavg_Chan_A2G);
        Gam_ms_A2G = 1/(exp(sigX_A2G^2)-1);
        Gam_ps_A2G = Pavg_Chan_A2G*sqrt((1+Gam_ms_A2G)/Gam_ms_A2G);
        
        fid = fopen(['C:\Code\Team\SoSGeneWithCuda\SoSGeneWithCuda\x64\Release\lognak',num2str(mm),'_',num2str(ii),'.bin'],'rb');
        H = fread(fid,inf,'float').';
        fclose(fid);
        Nak_Lognorm_H = Nak_Lognorm_H .* H;
%         Nak_Lognorm_H = Nak_Lognorm_H/sqrt(mean(Nak_Lognorm_H.^2)/(Gam_ps_G2A * Gam_ps_A2G));
        
        %% hu 接收机
        for nn = 1:length(RecvPower_G2)
%             Hc = randn(1,fs*endT,'single');
%             Hs = randn(1,fs*endT,'single');
%             
%             N0 = RecvPower_G2 * Gam_ps_G2A * Gam_ps_A2G/snr(nn);
%             N0_tmp = mean(Hc.^2)+mean(Hs.^2);
%             Noise = (Hc+1i*Hs)/sqrt(N0_tmp/N0);
%             
%             RecvSignal = LaunSignal * sqrt(RecvPower_G2) .* Nak_Lognorm_H + Noise;
%             RecvSignal = awgn(LaunSignal * sqrt(RecvPower_G2) .* Nak_Lognorm_H,snr_dB(nn),'measured');
            RecvSignal = LaunSignal * sqrt(RecvPower_G2(nn)) .* Nak_Lognorm_H + wgn(1,length(LaunSignal),N0,'complex','linear');
            switch modulation
                case 'BPSK'
                    DataDest = pskdemod(RecvSignal,2);
                case 'QPSK'
                    DataDest = pskdemod(RecvSignal,4);
                case '8PSK'
                    DataDest = pskdemod(RecvSignal,8);
                case '16QAM'
                    DataDest = qamdemod(RecvSignal,16);
                case '64QAM'
                    DataDest = qamdemod(RecvSignal,64);
            end
            [~,tmp] = symerr(DataSorce,DataDest);
            ser_sim(mm,nn) = ser_sim(mm,nn) + tmp;
             
%             if ii == 1 && nn == length(snr)
%                 %             figure;
%                 %             plot(abs(wgn(1,length(LaunSignal),N0,'complex','linear')));
% %                 scatterplot(LaunSignal);
% %                 grid on;
%                 %             RecvSignal = RecvSignal * sqrt(LaunPower_G1/(mean(abs(RecvSignal).^2)));
%                 scatterplot(RecvSignal);
%                 grid on;
%             end
        end
%         if ii == 1
%             ser_theory(mm,:) = CascadedGG_ser_BPSK(Nak_m_G2A,Nak_m_A2G,Gam_ms_G2A,Gam_ms_A2G,snr);
%         end
    end
end
ser_sim = ser_sim / n_times;

%% hu 画图
% hFig = figure;
% plot(snr_dB,ser_sim(1,:),SimulationColorList{1,1},snr_dB,ser_theory(1,:),TheoryColorList{1,1}, ...
%      snr_dB,ser_sim(2,:),SimulationColorList{1,2},snr_dB,ser_theory(2,:),TheoryColorList{1,2}, ...
%      snr_dB,ser_sim(3,:),SimulationColorList{1,3},snr_dB,ser_theory(3,:),TheoryColorList{1,3});
% zoom xon;
% xlabel('\itSNR\rm/dB');ylabel('ABER');
hFig = figure;
plot(h,ser_sim(1,:),SimulationColorList{1,1}, ...
     h,ser_sim(2,:),SimulationColorList{1,2}, ...
     h,ser_sim(3,:),SimulationColorList{1,3});
zoom xon;
xlabel('\it飞行高度\rm/m');ylabel('ABER');
   