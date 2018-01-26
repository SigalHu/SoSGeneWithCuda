function Nak_H = Gen_AutoCorNak_SimpFunc(fs,f_dopp,endT,Nak_m,Nak_power,mode)

if nargin == 5
    mode = 'real';
end

% clear all;clc;close all;
% endT = 20;    fs = 1000;
% f_dopp = 50; 
% Nak_m = 2.8;    Nak_power = 1;
% mode = 'comp';

if mode == 'real'
     Nak_H = zeros(1,endT*fs);
     Nak_var = Nak_power - (Nak_power/Nak_m)*(gamma(Nak_m+0.5)^2)/(gamma(Nak_m)^2);
     Gauss_var = Nak_var/(2*Nak_m - 2*gamma(Nak_m+0.5)^2/gamma(Nak_m)^2);
     if rem(2*Nak_m,1)==0                 %% 直接利用n个iid高斯过程平方和
       for n = 1:2*Nak_m
           [Ray_H] = Single_Rician_SoS(fs,f_dopp,0,0,endT); 
           A(n,1:endT*fs) = sqrt(Gauss_var)*real(Ray_H)*sqrt(2);
       end
       Nak_H = sqrt(sum(A.^2,1));
    else                                      %% 使用调整后的式子
       p = floor(2*Nak_m);
       if p == 0
           afi = 0;
       else
           afi = ( 2*p*Nak_m+sqrt(2*p*Nak_m*(p+1-2*Nak_m)) )/ ( p*(p+1) );
       end
       beta = 2*Nak_m - p*afi;
       for n = 1:p+1
           [Ray_H] = Single_Rician_SoS(fs,f_dopp,0,0,endT); 
           A(n,1:endT*fs) = sqrt(Gauss_var)*real(Ray_H)*sqrt(2);
       end
       Nak_H = sqrt(sum(A(1:p,:).^2,1)*afi + A(p+1,:).^2*beta); 
    end
else 
    Nak_m_quad = Nak_m/2;  Nak_power_quad = Nak_power/2;
    Nak_var = Nak_power_quad - (Nak_power_quad/Nak_m_quad)*(gamma(Nak_m_quad+0.5)^2)/(gamma(Nak_m_quad)^2);
    Gauss_var = Nak_var/(2*Nak_m_quad - 2*gamma(Nak_m_quad+0.5)^2/gamma(Nak_m_quad)^2);
    if rem(2*Nak_m_quad,1)==0                 %% 直接利用n个iid高斯过程平方和
       for n = 1:2*Nak_m_quad
           [Ray_H] = Single_Rician_SoS(fs,f_dopp,0,0,endT); 
           A_I(n,1:endT*fs) = sqrt(Gauss_var)*real(Ray_H)*sqrt(2);
           A_Q(n,1:endT*fs) = sqrt(Gauss_var)*imag(Ray_H)*sqrt(2);
       end
       tmp_Nak_I = sqrt(sum(A_I.^2,1));
       tmp_Nak_Q = sqrt(sum(A_Q.^2,1));
    else                                      %% 使用调整后的式子
       p = floor(2*Nak_m_quad);
       if p == 0
           afi = 0;
       else
           afi = ( 2*p*Nak_m_quad+sqrt(2*p*Nak_m_quad*(p+1-2*Nak_m_quad)) )/ ( p*(p+1) );
       end
       beta = 2*Nak_m_quad - p*afi;
       for n = 1:p+1
           [Ray_H] = Single_Rician_SoS(fs,f_dopp,0,0,endT); 
           A_I(n,1:endT*fs) = sqrt(Gauss_var)*real(Ray_H)*sqrt(2);
           A_Q(n,1:endT*fs) = sqrt(Gauss_var)*imag(Ray_H)*sqrt(2);
       end
       tmp_Nak_I = sqrt( sum(A_I(1:p,:).^2,1)*afi + A_I(p+1,:).^2*beta ); 
       tmp_Nak_Q = sqrt( sum(A_Q(1:p,:).^2,1)*afi + A_Q(p+1,:).^2*beta ); 
    end
    sign_I = randsrc(1,endT*fs);   sign_Q = randsrc(1,endT*fs);
    Nak_H = tmp_Nak_I.*sign_I + j*tmp_Nak_Q.*sign_Q;
end

% %%--confirm envelope pdf / phase pdf / m / power of complex Nakagami sequence--%%
% s = abs(Nak_H); 
% x = 0:0.05:max(s+1);  
% pdf_stat_en = hist(s,x);  
% pdf_ideal_en = ((2/gamma(Nak_m))*(Nak_m/Nak_power)^Nak_m)*x.^(2*Nak_m-1).*exp(-Nak_m*x.^2/Nak_power); 
% figure
% plot(x,pdf_ideal_en,x,pdf_stat_en/(length(s)*0.05),'*');    
% ylabel('Envelope PDF'),legend('Theroy','Sim');

