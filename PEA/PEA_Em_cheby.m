%% Housekeeping
clear
close all
clc
format long
addpath('../tools')

%% Set the stage
mypara;
min_lnA = log(0.6); max_lnA = log(1.4);
min_lnK = log(500); max_lnK = log(2500);
min_lnN = log(0.5); max_lnN = log(1.0);
degree = 7;
nA = 10;
nK = 10;
nN = 10;
damp_factor = 0.5;
maxiter = 10000;
tol = 1e-3;
options = optimoptions(@fsolve,'Display','none','Jacobian','off');

%% Grid creaton
[P,lnAgrid] = rouwen(rrho,0,ssigma/sqrt(1-rrho^2),nA);
P = P';
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

coeff_lnc = zeros(K,1);
coeff_lnv = zeros(K,1);
coeff_lnc(1) = log(c_ss);
coeff_lnv(1) = log(v_ss);

%% Create a initial guess from a rough PEA solution
if (exist('PEA_Em_cheby.mat','file')==2)
	load('PEA_Em_cheby.mat','coeff_lnmh','coeff_lnmf');
end
lnc_guess = zeros(N,1); lnv_guess = zeros(N,1);
parfor i = 1:N
    [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
    a = exp(lnAgrid(i_a)); k  = exp(lnKgrid(i_k)); n = exp(lnNgrid(i_n)); %#ok<PFBNS>

    
    lnEMH = ChebyshevND(degree,[lnAchebygrid(i_a),lnKchebygrid(i_k),lnNchebygrid(i_n)])*coeff_lnmh;
    lnEMF = ChebyshevND(degree,[lnAchebygrid(i_a),lnKchebygrid(i_k),lnNchebygrid(i_n)])*coeff_lnmf;
    c_guess = 1/(bbeta*exp(lnEMH));
    q_guess = kkappa/c_guess/(bbeta*exp(lnEMF));
    ttheta_guess = (q_guess/xxi)^(eeta-1);
    v_guess = ttheta_guess*(1-exp(lnNgrid(i_n)));
    lnc_guess(i) = log(c_guess);
    lnv_guess(i) = log(v_guess);
end
temp_lnc = X'*lnc_guess;
temp_lnv = X'*lnv_guess;
temp_X = X'*X;
parfor i_term = 1:K
    coeff_lnc(i_term) = temp_lnc(i_term)/temp_X(i_term,i_term);
    coeff_lnv(i_term) = temp_lnv(i_term)/temp_X(i_term,i_term);
end

lnpolicy_new = zeros(N,2);

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
        state = [lnAgrid(i_a),lnKgrid(i_k),lnNgrid(i_n),tot_stuff(i),ustuff(i),i_a];
        lnc_guess = ChebyshevND(degree,[lnAchebygrid(i_a),lnKchebygrid(i_k),lnNchebygrid(i_n)])*coeff_lnc;
        lnv_guess = ChebyshevND(degree,[lnAchebygrid(i_a),lnKchebygrid(i_k),lnNchebygrid(i_n)])*coeff_lnv;
        x0 = [lnc_guess,lnv_guess];
        [lnpolicy_new(i,:),fval,exitflag] = nested_timeiter_obj(state,param,coeff_lnc,coeff_lnv,lnAgrid,lnAchebygrid,P,nA,x0,options);
    end
    coeff = (X'*X)\(X'*lnpolicy_new);
    coeff_lnc_temp = coeff(:,1); coeff_lnv_temp = coeff(:,2);
    
    %% Damped update
    coeff_lnc_new = (1-damp_factor)*coeff_lnc_temp+(damp_factor)*coeff_lnc;
    coeff_lnv_new = (1-damp_factor)*coeff_lnv_temp+(damp_factor)*coeff_lnv;
    
    %% Compute norm
    diff = norm([coeff_lnc;coeff_lnv]-[coeff_lnc_new;coeff_lnv_new],Inf);
    
    %% Update
    coeff_lnc = coeff_lnc_new;
    coeff_lnv = coeff_lnv_new;
    iter = iter+1;
    %% Display something
    iter
    diff
    coeff_lnc;
    coeff_lnv;

end;

%% Euler equation error
nk = 10; nA = 10; nnn = 10;
lnKgrid_ee = log(linspace(1100,1700,nk));
lnAgrid_ee = log(linspace(0.8,1.2,nA));
lnNgrid_ee = log(linspace(0.9,0.999,nnn));
EEerror_c = 999999*ones(nA,nk,nnn);
EEerror_v = 999999*ones(nA,nk,nnn);
      
parfor i = 1:nA*nk*nnn
    [i_A,i_k,i_n] = ind2sub([nA nk nnn],i);
    lnk = lnKgrid_ee(i_k);
    lnkcheby = -1 + 2*(lnk-min_lnK)/(max_lnK-min_lnK);
    lna = lnAgrid_ee(i_A);
    lnacheby = -1 + 2*(lna-min_lnA)/(max_lnA-min_lnA);
    lnn = lnNgrid_ee(i_n);
    lnncheby = -1 + 2*(lnn-min_lnN)/(max_lnN-min_lnN);
    a = exp(lna); k  = exp(lnk); n = exp(lnn);
    tot_stuff = a*k^aalpha*n^(1-aalpha) + (1-ddelta)*k + z*(1-n);
    ustuff = xxi*(1-n)^(1-eeta);
    lnc = ChebyshevND(degree,[lnacheby,lnkcheby,lnncheby])*coeff_lnc;
    lnv = ChebyshevND(degree,[lnacheby,lnkcheby,lnncheby])*coeff_lnv;
    c = exp(lnc);
    v = exp(lnv);
    kplus = tot_stuff - c - kkappa*v;
    nplus = (1-x)*exp(lnn) + xxi*v^(eeta)*(1-n)^(1-eeta);
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
        lnc_plus = ChebyshevND(degree,[lnaplus_cheby,lnkplus_cheby,lnnplus_cheby])*coeff_lnc;
        lnv_plus = ChebyshevND(degree,[lnaplus_cheby,lnkplus_cheby,lnnplus_cheby])*coeff_lnv;
        cplus = exp(lnc_plus);
        vplus = exp(lnv_plus);
        tthetaplus = vplus/(1-nplus);
        qplus = xxi*tthetaplus^(eeta-1);
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
EEerror_c_inf = norm(EEerror_c(:),inf)
EEerror_v_inf = norm(EEerror_v(:),inf)

EEerror_c_mean = mean(EEerror_c(:))
EEerror_v_mean = mean(EEerror_v(:))



