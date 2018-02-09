% clear all;  close all; clc;

n_times = 1000;
modulation = 'BPSK'; % BPSK QPSK 8PSK 16QAM 64QAM
snr_dB_min = 3;
d_G2A = 0;
h = 200:40:1e3;

ParaList = [10.3 3.2; ...    % hu 丘陵
            9.2  3.9; ...    % hu 山区
            9.6  4.2];       % hu 海洋

SimulationColorList = {'r.:','b.:','k.:','g.:';
                       'r*:','b*:','k*:','g*:';
                       'ro:','bo:','ko:','go:';};
TheoryColorList = {'r-','b-','k-','g-'};

fc = 968e6;
fs = 1000;
endT = 100;
fd = 48.4;

LaunPower_dB = 0;
LaunAnte_dB = 0;
RecvAnte_dB = 0;

PathLen = sqrt(d_G2A^2+h.^2);
Pavg_Chan = 1;

% hu 理论表达式
PathLoss_Calc = @(d,f) 32.44 + 20*log10(d/1e3) + 20*log10(f/1e6);

ser_sim = zeros(size(ParaList,1),length(h));

for ii=1:n_times
    %% hu 发射机
    LaunPower = 10^(LaunPower_dB/10);
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
    LaunSignal = LaunSignal * sqrt(LaunPower/(mean(abs(LaunSignal).^2)));
    
    %% hu 信道
    % hu 确定值部分
    PathLoss_dB = PathLoss_Calc(PathLen,fc);
    RecvPower_dB = LaunPower_dB + LaunAnte_dB + RecvAnte_dB - PathLoss_dB;
    RecvPower = 10.^(RecvPower_dB/10);
    
    % hu 随机值部分
    % hu 产生噪声
    Shadow_dB = min(ParaList(:,2));
    sigX = Shadow_dB/8.686;
    miuX = log(Pavg_Chan);
    Gam_ms = 1/(exp(sigX^2)-1);
    Gam_ps = Pavg_Chan*sqrt((1+Gam_ms)/Gam_ms);
    
    snr_min = 10^(snr_dB_min/10);
    N0 = min(RecvPower * Gam_ps)/snr_min;
    
    for mm = 1:size(ParaList,1)
        Nak_m = ParaList(mm,1);
        Shadow_dB = ParaList(mm,2);

        sigX = Shadow_dB/8.686;
        miuX = log(Pavg_Chan);
        Gam_ms = 1/(exp(sigX^2)-1);
        Gam_ps = Pavg_Chan*sqrt((1+Gam_ms)/Gam_ms);

        fid = fopen(['C:\Code\Team\SoSGeneWithCuda\SoSGeneWithCuda\x64\Release\lognak',num2str(mm),'_',num2str(ii),'.bin'],'rb');
        Nak_Lognorm_H = fread(fid,inf,'float').';
        fclose(fid);

        %% hu 接收机
        for nn = 1:length(RecvPower)
            RecvSignal = LaunSignal * sqrt(RecvPower(nn)) .* Nak_Lognorm_H + wgn(1,length(LaunSignal),N0,'complex','linear');
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
        end
    end
end
ser_sim = ser_sim / n_times;

%% hu 画图
hFig = figure;
plot(h,ser_sim(1,:),SimulationColorList{1,1}, ...
     h,ser_sim(2,:),SimulationColorList{1,2}, ...
     h,ser_sim(3,:),SimulationColorList{1,3});
zoom xon;
xlabel('\it飞行高度\rm/m');ylabel('ABER');
   