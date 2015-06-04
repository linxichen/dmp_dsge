%% Housekeeping
clear
close all
clc
format long
addpath('../tools')

%% Set the stage
mypara;
min_lnA = log(0.5); max_lnA = log(1.7);
min_lnK = log(100); max_lnK = log(3000);
min_lnN = log(0.1); max_lnN = log(0.9999);
degree = 7;
nA = 10;
nK = 10;
nN = 10;
damp_factor = 0.2;
maxiter = 10000;
tol = 1.5e-7;
options = optimoptions(@fsolve,'Display','final-detailed','Jacobian','off');
[epsi_nodes,weight_nodes] = GH_nice(9,0,1);
n_nodes = length(epsi_nodes);

%% Grid creaton
lnAgrid = ChebyshevRoots(nA,'Tn',[log(0.85),log(1.15)]);
lnKgrid = ChebyshevRoots(nK,'Tn',[min_lnK,max_lnK]);
lnNgrid = ChebyshevRoots(nN,'Tn',[min_lnN,max_lnN]);
lnAchebygrid = ChebyshevRoots(nA,'Tn');
lnKchebygrid = ChebyshevRoots(nK,'Tn');
lnNchebygrid = ChebyshevRoots(nN,'Tn');
N = nA*nK*nN;

[fakebasis,order_table] = ChebyshevND(degree,[0,0,0]);
K = size(fakebasis,2);

%% Encapsulate all parameters
param = [... 
 bbeta; % 1
 ggamma; % 2
 kkappa; % 3
 eeta; % 4
 rrho; %5
 ssigma; %6
 min_lnA;  %7
 max_lnA; %8
 min_lnK; %9
 max_lnK; %10
 min_lnN; % 11
 max_lnN; % 12
 degree; % 13
 x; % 14
 aalpha; % 15
 ddelta; % 16
 xxi; % 17
 ttau; % 18
 z
 ];

%% Precomputation
X = zeros(N,K);
tot_stuff = zeros(N,1); ustuff = zeros(N,1);
parfor i = 1:N
    [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
    a = exp(lnAgrid(i_a)); k  = exp(lnKgrid(i_k)); n = exp(lnNgrid(i_n)); %#ok<PFBNS>
    tot_stuff(i) = a*k^aalpha*n^(1-aalpha) + (1-ddelta)*k + z*(1-n);
    ustuff(i) = xxi*(1-n)^(1-eeta);
    X(i,:) = ChebyshevND(degree,[lnAchebygrid(i_a),lnKchebygrid(i_k),lnNchebygrid(i_n)])
end

coeff_lnmh = zeros(1,K);
coeff_lnmf = zeros(1,K);
%% Create a initial guess from a rough PEA solution
if (exist('PEA_Em_cheby.mat','file')==2)
	load('PEA_Em_cheby.mat','coeff_lnmh','coeff_lnmf');
end

lnEM_new = zeros(N,2);

%% Solve for SS
kss = k_ss;
nss = n_ss;

%% Iteration
opts = statset('nlinfit');
%opts.RobustWgtFun = 'bisquare';
opts.Display = 'final';
opts.MaxIter = 10000;
diff = 10; iter = 0;
while (diff>tol && iter <= maxiter)
    %% Fixed point iter step, find EMF EMH that solve euler exactly
    parfor i = 1:N
        [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
        state = [lnAgrid(i_a),lnKgrid(i_k),lnNgrid(i_n),tot_stuff(i),ustuff(i)];
        lnEMH = ChebyshevND(degree,[lnAchebygrid(i_a),lnKchebygrid(i_k),lnNchebygrid(i_n)])*coeff_lnmh;
        lnEMF = ChebyshevND(degree,[lnAchebygrid(i_a),lnKchebygrid(i_k),lnNchebygrid(i_n)])*coeff_lnmf;
        c = 1/(bbeta*exp(lnEMH));
        q = kkappa/c/(bbeta*exp(lnEMF));
        v = (q/ustuff(i))^(1/(eeta-1));
        kplus = tot_stuff(i) - c - kkappa*v;
        nplus = (1-x)*exp(lnNgrid(i_n)) + q*v;
        lnkplus = log(kplus); lnnplus = log(nplus);
        lnkplus_cheby = -1 + 2*(lnkplus-min_lnK)/(max_lnK-min_lnK);
        lnnplus_cheby = -1 + 2*(lnnplus-min_lnN)/(max_lnN-min_lnN);
        if (lnkplus_cheby < -1 || lnkplus_cheby > 1)
            lnkplus
            error('kplus out of bound')
        end
        if (lnnplus_cheby < -1 || lnnplus_cheby > 1)
            lnnplus_cheby
            lnnplus
            error('nplus out of bound')
        end
        
        % Find expected mh, mf tomorrow if current coeff applies tomorrow
        EMH_hat = 0;
        EMF_hat = 0;
        for i_node = 1:n_nodes
            eps = epsi_nodes(i_node);
            lnaplus = rrho*lnAgrid(i_a) + ssigma*eps;
            lnaplus_cheby = -1 + 2*(lnaplus-min_lnA)/(max_lnA-min_lnA);
            if (lnaplus_cheby < -1 || lnaplus_cheby > 1)
				lnAgrid(i_a)
				lnaplus
                error('Aplus out of bound')
            end
            lnEMH_plus = ChebyshevND(degree,[lnaplus_cheby,lnkplus_cheby,lnnplus_cheby])*coeff_lnmh;
            lnEMF_plus = ChebyshevND(degree,[lnaplus_cheby,lnkplus_cheby,lnnplus_cheby])*coeff_lnmf;
            cplus = 1/(bbeta*exp(lnEMH_plus));
            qplus = kkappa/cplus/(bbeta*exp(lnEMF_plus));
            tthetaplus = (qplus/xxi)^(1/(eeta-1));
            EMH_hat = EMH_hat + weight_nodes(i_node)*((1-ddelta+aalpha*exp(lnaplus)*(kplus/nplus)^(aalpha-1))/cplus);
            EMF_hat = EMF_hat + weight_nodes(i_node)*(( (1-ttau)*((1-aalpha)*exp(lnaplus)*(kplus/nplus)^aalpha-z-ggamma*cplus) + (1-x)*kkappa/qplus - ttau*kkappa*tthetaplus )/cplus );
        end        
        lnEM_new(i,:) = [log(EMH_hat),log(EMF_hat)];
    end
    coeff = (X'*X)\(X'*lnEM_new);
    coeff_lnmh_temp = coeff(:,1); coeff_lnmf_temp = coeff(:,2);
    
    %% Damped update
    coeff_lnmh_new = (1-damp_factor)*coeff_lnmh_temp+(damp_factor)*coeff_lnmh;
    coeff_lnmf_new = (1-damp_factor)*coeff_lnmf_temp+(damp_factor)*coeff_lnmf;
    
    %% Compute norm
    diff = norm([coeff_lnmh;coeff_lnmf]-[coeff_lnmh_new;coeff_lnmf_new],Inf);
    
    %% Update
    coeff_lnmh = coeff_lnmh_new;
    coeff_lnmf = coeff_lnmf_new;
    iter = iter+1;
    %% Display something
    iter
    diff
    coeff_lnmh;
    coeff_lnmf;

end;

%% Euler equation error
nk = 10; nA = 10; nnn = 10;
lnKgrid = log(linspace(0.8*kss,1.2*kss,nk));
lnAgrid = log(linspace(0.8,1.2,nA));
lnNgrid = log(linspace(0.9,0.999,nnn));
EEerror_c = 999999*ones(nA,nk,nnn);
EEerror_v = 999999*ones(nA,nk,nnn);
      
parfor i = 1:nA*nk*nnn
    [i_A,i_k,i_n] = ind2sub([nA nk nnn],i);
    lnk = lnKgrid(i_k);
    lnkcheby = -1 + 2*(lnk-min_lnK)/(max_lnK-min_lnK);
    lna = lnAgrid(i_A);
    lnacheby = -1 + 2*(lna-min_lnA)/(max_lnA-min_lnA);
    lnn = lnNgrid(i_n);
    lnncheby = -1 + 2*(lnn-min_lnN)/(max_lnN-min_lnN);
    a = exp(lna); k  = exp(lnk); n = exp(lnn);
    tot_stuff = a*k^aalpha*n^(1-aalpha) + (1-ddelta)*k + z*(1-n);
    ustuff = xxi*(1-n)^(1-eeta);
    lnEMH = ChebyshevND(degree,[lnacheby,lnkcheby,lnncheby])*coeff_lnmh;
    lnEMF = ChebyshevND(degree,[lnacheby,lnkcheby,lnncheby])*coeff_lnmf;
    c = 1/(bbeta*exp(lnEMH));
    q = kkappa/c/(bbeta*exp(lnEMF));
    v = (q/ustuff)^(1/(eeta-1));
    kplus = tot_stuff - c - kkappa*v;
    nplus = (1-x)*exp(lnn) + q*v;
    lnkplus = log(kplus); lnnplus = log(nplus);
    lnkplus_cheby = -1 + 2*(lnkplus-min_lnK)/(max_lnK-min_lnK);
    lnnplus_cheby = -1 + 2*(lnnplus-min_lnN)/(max_lnN-min_lnN);
    
    % Find expectations
    EMH_hat = 0;
    EMF_hat = 0;
    for i_node = 1:n_nodes
        eps = epsi_nodes(i_node);
        lnaplus = rrho*lna + ssigma*eps;
        lnaplus_cheby = -1 + 2*(lnaplus-min_lnA)/(max_lnA-min_lnA);
        if (lnaplus_cheby < -1 || lnaplus_cheby > 1)
            error('Aplus out of bound')
        end
        lnEMH_plus = ChebyshevND(degree,[lnaplus_cheby,lnkplus_cheby,lnnplus_cheby])*coeff_lnmh;
        lnEMF_plus = ChebyshevND(degree,[lnaplus_cheby,lnkplus_cheby,lnnplus_cheby])*coeff_lnmf;
        cplus = 1/(bbeta*exp(lnEMH_plus));
        qplus = kkappa/cplus/(bbeta*exp(lnEMF_plus));
        tthetaplus = (qplus/xxi)^(1/(eeta-1));
        EMH_hat = EMH_hat + weight_nodes(i_node)*((1-ddelta+aalpha*exp(lnaplus)*(kplus/nplus)^(aalpha-1))/cplus);
        EMF_hat = EMF_hat + weight_nodes(i_node)*(( (1-ttau)*((1-aalpha)*exp(lnaplus)*(kplus/nplus)^aalpha-z-ggamma*cplus) + (1-x)*kkappa/qplus - ttau*kkappa*tthetaplus )/cplus );
    end
    c_imp = 1/(bbeta*EMH_hat);
    q_imp = kkappa/c_imp/(bbeta*EMF_hat);
    ttheta_imp = (q_imp/xxi)^(1/(eeta-1));
    v_imp = ttheta_imp*(1-n);
    
    EEerror_c(i) = abs((c-c_imp)/c_ss);
    EEerror_v(i) = abs((v-v_imp)/v_ss);
    
end
EEerror_c_inf = norm(EEerror_c(:),inf);
EEerror_v_inf = norm(EEerror_v(:),inf);

EEerror_c_mean = mean(EEerror_c(:));
EEerror_v_mean = mean(EEerror_v(:));

figure
plot(lnKgrid,EEerror_c(ceil(nA/2),:,ceil(nnn/2)))

figure
plot(lnKgrid,EEerror_v(ceil(nA/2),:,ceil(nnn/2)))
xlabel('log(k)')
ylabel('Error in vacancy')

figure
plot(lnNgrid,squeeze(EEerror_v(ceil(nA/2),ceil(nA/2),:)))
xlabel('log(n)')
ylabel('Error in vacancy')
%% Implied policy functions and find wages
lnAgrid = csvread('../CUDA_VFI/results/Agrid.csv');
lnKgrid = csvread('../CUDA_VFI/results/Kgrid.csv');
lnNgrid = csvread('../CUDA_VFI/results/Ngrid.csv');
nA = length(lnAgrid);
nk = length(lnKgrid);
nnn = length(lnNgrid);

kk = zeros(nA,nk,nnn);
cc = kk;
vv = kk;
nn = kk;
ttheta_export = kk;
wage_export = kk;
cc_dynare = kk;
kk_dynare = kk;
nn_dynare = kk;
vv_dynare = kk;

mmummu = kk;
for i_k = 1:nk
    for i_n = 1:nnn
        for i_A = 1:nA
            state(1) = lnAgrid(i_A); lna = state(1);
            state(2) = lnKgrid(i_k); lnk = state(2);
            state(3) = lnNgrid(i_n); lnn = state(3);
            EM = exp([1 log(state)]*[coeff_mh coeff_mf]);
            
            y = lna*(lnk)^(aalpha)*(lnn)^(1-aalpha);
            c = (bbeta*EM(1))^(-1);
            ttheta = (kkappa/(c*xxi*bbeta*EM(2)))^(1/(eeta-1));
            v = ttheta*(1-lnn);
            mh = (1-ddelta+aalpha*y/lnk)/c;
            mf = ( (1-ttau)*((1-aalpha)*y/lnn-z-ggamma*c) + (1-x)*kkappa/xxi*ttheta^(1-eeta) - ttau*kkappa*ttheta )/c;
            w = ttau*lna*lnk^(aalpha)*(1-aalpha)*lnn^(-aalpha) + (1-ttau)*(z+ggamma*c) + ttau*kkappa*ttheta;
    
            kk(i_A,i_k,i_n) = y - c +(1-ddelta)*lnk - kkappa*v + z*(1-nn(i_A,i_k,i_n));
            nn(i_A,i_k,i_n) = (1-x)*lnn + xxi*ttheta^(eeta)*(1-lnn);
            cc(i_A,i_k,i_n) = c;
            vv(i_A,i_k,i_n) = v;
            
            cc_dynare(i_A,i_k,i_n) = exp(2.111091 + 0.042424/rrho*log(lnAgrid(i_A))/ssigma + 0.615500*(log(lnKgrid(i_k))-log(k_ss)) + 0.014023*(log(lnNgrid(i_n))-log(n_ss)) );
            kk_dynare(i_A,i_k,i_n) = exp(7.206845 + 0.006928/rrho*log(lnAgrid(i_A))/ssigma + 0.997216*(log(lnKgrid(i_k))-log(k_ss)) + 0.005742*(log(lnNgrid(i_n))-log(n_ss)) );
            nn_dynare(i_A,i_k,i_n) = exp(-0.056639 + 0.011057/rrho*log(lnAgrid(i_A))/ssigma + 0.001409*(log(lnKgrid(i_k))-log(k_ss)) + 0.850397*(log(lnNgrid(i_n))-log(n_ss)) );
            
            % Export prices
            wage_export(i_A,i_k,i_n) = w;
            ttheta_export(i_A,i_k,i_n) = ttheta;
        end
    end
end
save('PEA_Em.mat');


i_mid_n = ceil(nnn/2);
i_mid_A = ceil(nA/2);
linewitdh=1.5;
figure
plot(lnKgrid,squeeze(kk(i_mid_A,:,i_mid_n)),lnKgrid,squeeze(kk_dynare(i_mid_A,:,i_mid_n)),'LineWidth',linewitdh)
axis('tight')
xlabel('k(t)')
ylabel('k(t+1)')
legend('Nonlinear','Linear')

figure
plot(lnKgrid,squeeze(nn(i_mid_A,:,i_mid_n)),lnKgrid,squeeze(nn_dynare(i_mid_A,:,i_mid_n)),'LineWidth',linewitdh)
axis('tight')
xlabel('k(t)')
ylabel('n(t+1)')
legend('Nonlinear','Linear')

figure
plot(lnKgrid,squeeze(cc(i_mid_A,:,i_mid_n)),lnKgrid,squeeze(cc_dynare(i_mid_A,:,i_mid_n)),'LineWidth',linewitdh)
axis('tight')
xlabel('k(t)')
ylabel('c(t)')
legend('Nonlinear','Linear')

figure
plot(lnKgrid,squeeze(wage_export(i_mid_A,:,i_mid_n)),'LineWidth',linewitdh)
axis('tight')
xlabel('k(t)')
ylabel('wage')
legend('Nonlinear')

figure
plot(lnKgrid,squeeze(ttheta_export(i_mid_A,:,i_mid_n)),'LineWidth',linewitdh)
axis('tight')
xlabel('k(t)')
ylabel('Tightness')
legend('Nonlinear')

