clear
close all
clc
addpath('../../PEA/')
format long

%% load stuff
Agrid = csvread('Agrid.csv'); nA = length(Agrid);
Kgrid = csvread('Kgrid.csv'); nk = length(Kgrid);
Ngrid = csvread('Ngrid.csv'); nnn = length(Ngrid);
copt = csvread('copt.csv'); vopt = csvread('vopt.csv'); nopt = csvread('nopt.csv');
copt = reshape(copt,[nA nk nnn]);
vopt = reshape(vopt,[nA nk nnn]);
nopt = reshape(nopt,[nA nk nnn]);
koptind = csvread('koptind.csv');
kopt = Kgrid(koptind+1); % zero based to one based
kopt = reshape(kopt,[nA nk nnn]);
mypara;
plot(Kgrid,kopt(ceil(nA/2),:,ceil(nnn/2)))

%% Compare consumption
cPEA = csvread('../cPEA_export.csv');
cPEA = reshape(cPEA,[nA,nk,nnn]);
h = figure;
plot(Kgrid,copt(ceil(nA/2),:,ceil(nnn/2)),Kgrid,cPEA(ceil(nA/2),:,ceil(nnn/2)))
legend('VFI','PEA')
print(h,'c_comparison.eps','-depsc2')


%% Compare capital
h = figure;
kPEA = csvread('../kPEA_export.csv');
kPEA = reshape(kPEA,[nA,nk,nnn]);
plot(Kgrid,kopt(ceil(nA/2),:,ceil(nnn/2))-(1-ddelta)*Kgrid',Kgrid,kPEA(ceil(nA/2),:,ceil(nnn/2))-(1-ddelta)*Kgrid')
legend('VFI','PEA')
print(h,'k_comparison.eps','-depsc2')

%% Compare labor tmr
h = figure;
nPEA = csvread('../nPEA_export.csv');
nPEA = reshape(nPEA,[nA,nk,nnn]);
plot(Kgrid,nopt(ceil(nA/2),:,ceil(nnn/2)),Kgrid,nPEA(ceil(nA/2),:,ceil(nnn/2)))
legend('VFI','PEA')
print(h,'n_comparison.eps','-depsc2')

%% Find implied wage and ttheta
wage_vfi = zeros(nA,nk,nnn);
ttheta_vfi = zeros(nA,nk,nnn);
for i_A = 1:nA
    for i_k = 1:nk
        for i_n = 1:nnn
            ttheta_vfi(i_A,i_k,i_n) = vopt(i_A,i_k,i_n)/(1-Ngrid(i_n));
            wage_vfi(i_A,i_k,i_n) = ttau*Agrid(i_A)*(1-aalpha)*Kgrid(i_k)^(aalpha)*Ngrid(i_n)^(-aalpha)+(1-ttau)*(z+ggamma*copt(i_A,i_k,i_n))+ttau*kkappa*vopt(i_A,i_k,i_n)/(1-Ngrid(i_n));
        end
    end
end

%% Compare wage
h = figure;
wagePEA = csvread('../wage_export.csv');
wagePEA = reshape(wagePEA,[nA,nk,nnn]);
plot(Kgrid,wage_vfi(ceil(nA/2),:,ceil(nnn/2)),Kgrid,wagePEA(ceil(nA/2),:,ceil(nnn/2)))
legend('VFI','PEA')
print(h,'wage_comparison.eps','-depsc2')

%% Compare ttheta
h = figure;
tthetaPEA = csvread('../ttheta_export.csv');
tthetaPEA = reshape(tthetaPEA,[nA,nk,nnn]);
plot(Kgrid,ttheta_vfi(ceil(nA/2),:,ceil(nnn/2)),Kgrid,tthetaPEA(ceil(nA/2),:,ceil(nnn/2)))
legend('VFI','PEA')
print(h,'ttheta_comparison.eps','-depsc2')

