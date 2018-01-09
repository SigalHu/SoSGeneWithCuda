clear all;  close all; clc;

path_num = [3 4 5 6 7 8 9 10];
cpu_time = [34.9 53.8 100.7 170.3 304.2 597.9 1221.7 2441.1]/1000;
gpu_time = [168.7 174.2 178.9 181.4 201 223 249.8 288]/1000;

figure;
bar(path_num,[gpu_time;cpu_time].');

xlabel('$n$','Interpreter','latex');
ylabel('‘ÀÀ„ ±º‰/s','Interpreter','latex');
grid on;zoom xon;
