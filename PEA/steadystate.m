function ss = steadystate(input)
load DMP_RBC_PARAS.mat;
k = input(1);
n = input(2);
c = input(3);
ttheta = input(4);
v = input(5);

% Aux variables
q = xxi*ttheta^(eeta-1);
mmu = ttheta*q;

ss(1) = 1 - bbeta*(1-ddelta+aalpha*(k/n)^(aalpha-1));
ss(2) = kkappa/q - bbeta*( (1-ttau)*((1-aalpha)*(k/n)^(aalpha)-z-ggamma*c) + (1-x)*kkappa/q - ttau*kkappa*ttheta );
ss(3) = c + ddelta*k + kkappa*v - k^(aalpha)*n^(1-aalpha);
ss(4) = ttheta - v/(1-n);
ss(5) = x*n - mmu*(1-n);

end