% close all;  clear all;  clc;

Nak_m = [0.5 1 5 10];
r_dB = -30:0.5:10;
r = 10.^(r_dB/20);%0.001:0.1:10;%
fd = 500;
for i = 1:4
%     LCR(i,:) = sqrt(2*pi)* fd * Nak_m(i)^(Nak_m(i)-0.5) * r.^(2*Nak_m(i)-1) .* exp(-Nak_m(i)*r.^2) / gamma(Nak_m(i));
    AFD(i,:) = gamma(Nak_m(i))*gammainc(Nak_m(i)*r.^2,Nak_m(i))./(sqrt(2*pi)* fd * Nak_m(i)^(Nak_m(i)-0.5)*r.^(2*Nak_m(i)-1).*exp(-Nak_m(i)*r.^2));
end
% figure
% semilogy(r_dB,LCR/fd);   
% xlabel('���޵�ƽr(dB)'); ylabel('��ƽͨ����N(r)')
% legend('m=0.5','m=1','m=5','m=10');

figure
semilogy(r_dB,AFD*fd);   
xlabel('���޵�ƽr'); ylabel('ƽ��˥�����ʱ��T(r)*f_d')
legend('m=0.5','m=1','m=5','m=10');
