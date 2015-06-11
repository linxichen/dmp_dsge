%% Housekeeping
clear
close all
clc
format long
addpath('../tools')

%% Set the stage
mypara;
nA = 10;
nK = 11;
nN = 12;
[P,lnAgrid] = rouwen(rrho,0,ssigma/sqrt(1-rrho^2),nA);
P = P';
min_lnA = lnAgrid(1); max_lnA = lnAgrid(end);
min_lnK = log(900); max_lnK = log(2500);
min_lnN = log(0.8); max_lnN = log(0.99);
degree = 9;
damp_factor = 0.0;
maxiter = 10000;
tol = 1e-10;
options = optimoptions(@fsolve,'Display','none','Jacobian','off');

%% Grid creaton
lnKgrid = ChebyshevRoots(nK,'Tn',[min_lnK,max_lnK]);
lnNgrid = ChebyshevRoots(nN,'Tn',[min_lnN,max_lnN]);
lnAchebygrid = ChebyshevRoots(nA,'Tn');
lnKchebygrid = ChebyshevRoots(nK,'Tn');
lnNchebygrid = ChebyshevRoots(nN,'Tn');
N = nA*nK*nN;
[Kmesh,Amesh,Nmesh] = meshgrid(lnKgrid,lnAgrid,lnNgrid);


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

%% Create a initial guess from a rough PEA solution
coeff_lnmh = zeros(K,1); coeff_lnmf = zeros(K,1);
coeff_lnmh(1) = 2.25112947095552;
coeff_lnmh(2) = -0.00422995273814383;
coeff_lnmh(3) = -0.608159319666493;
coeff_lnmh(4) = -0.0414379097902152;

coeff_lnmf(1) = 2.37578644280901;
coeff_lnmf(2) = -0.0371929066636178;
coeff_lnmf(3) = -0.367492859368021;
coeff_lnmf(4) = 2.17598356908754;


% if (exist('PEA_Em_cheby.mat','file')==2)
%     load('PEA_Em_cheby.mat','coeff_lnmh','coeff_lnmf')
% else
%     coeff_mh = [2.197278872016918; -0.030892629079668; -0.581445054648990; -0.004225383144729]; % one constant, each for state variable
%     coeff_mf = [2.281980399764238; 1.729203578753512; -0.315489670998162; -0.115805845378316];
% end
% lnEmh_train = zeros(N,1); lnEmf_train = zeros(N,1);
% parfor i = 1:N
%     [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
%     lnEmh_train(i) = ([1 lnAgrid(i_a) lnKgrid(i_k) lnNgrid(i_n)]*coeff_mh);
%     lnEmf_train(i) = ([1 lnAgrid(i_a) lnKgrid(i_k) lnNgrid(i_n)]*coeff_mf)
% end
% coeff_lnmh = (X'*X)\(X'*(lnEmh_train));
% coeff_lnmf = (X'*X)\(X'*(lnEmf_train));
% coeff_lnmh_old = coeff_lnmh;
% coeff_lnmf_old = coeff_lnmf;

lnEM_new = zeros(N,2);
lnKgrid_endo = zeros(N,1);
lnNgrid_endo = zeros(N,1);
lnAgrid_endo = zeros(N,1);
lnEMH_endo = zeros(N,1);
lnEMF_endo = zeros(N,1);

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
    %% Time iter step, uses endo grid technique
    parfor i = 1:N
        [i_a,i_kplus,i_nplus] = ind2sub([nA,nK,nN],i);
        nplus = exp(lnNgrid(i_nplus)); kplus = exp(lnKgrid(i_kplus));
        EMH = 0; EMF = 0;
        for i_node = 1:nA
            lnaplus = lnAgrid(i_node);
            lnaplus_cheby = lnAchebygrid(i_node);
            lnEMH_plus = ChebyshevND(degree,[lnaplus_cheby,lnKchebygrid(i_kplus),lnNchebygrid(i_nplus)])*coeff_lnmh;
            lnEMF_plus = ChebyshevND(degree,[lnaplus_cheby,lnKchebygrid(i_kplus),lnNchebygrid(i_nplus)])*coeff_lnmf;
            cplus = 1/(bbeta*exp(lnEMH_plus));
            qplus = kkappa/cplus/(bbeta*exp(lnEMF_plus));
            tthetaplus = (qplus/xxi)^(1/(eeta-1));
            EMH = EMH + P(i_a,i_node)*((1-ddelta+aalpha*exp(lnaplus)*(kplus/nplus)^(aalpha-1))/cplus);
            EMF = EMF + P(i_a,i_node)*(( (1-ttau)*((1-aalpha)*exp(lnaplus)*(kplus/nplus)^aalpha-z-ggamma*cplus) + (1-x)*kkappa/qplus - ttau*kkappa*tthetaplus )/cplus );
        end
        c = 1/(bbeta*EMH);
        q = kkappa/c/(bbeta*EMF);
        ttheta = (q/xxi)^(eeta-1);
        n = (nplus-q*ttheta)/(1-x-q*ttheta);
        v = ttheta*(1-n);
        findk = @(k) exp(lnAgrid(i_a))*k^aalpha*n^(1-aalpha)+z*(1-n)+(1-ddelta)*k-kkappa*v-c-kplus;
        x0 = kplus;
        [k,fval,exitflag] = fsolve(findk,x0,options);
        
        lnKgrid_endo(i) = log(k); lnNgrid_endo(i) = log(n); lnAgrid_endo(i) = lnAgrid(i_a);
        lnEMH_endo(i) = log(EMH); lnEMF_endo(i) = log(EMF);     
    end
    
    %% Interpolation to find values at
    F_EMH = scatteredInterpolant(lnAgrid_endo,lnKgrid_endo,lnNgrid_endo,lnEMH_endo,'natural','linear');
    F_EMF = scatteredInterpolant(lnAgrid_endo,lnKgrid_endo,lnNgrid_endo,lnEMF_endo,'natural','linear');
    lnEMH_new = F_EMH(Amesh,Kmesh,Nmesh);
    lnEMF_new = F_EMF(Amesh,Kmesh,Nmesh);
    coeff_lnmh_temp = (X'*X)\(X'*lnEMH_new(:)); coeff_lnmf_temp = (X'*X)\(X'*lnEMF_new(:));
    
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
nk = 10; nnn = 10;
Kgrid = linspace(1100,1500,nk);
Agrid = exp(lnAgrid);
Ngrid = linspace(0.9,0.97,nnn);
EEerror_c = 999999*ones(nA,nk,nnn);
EEerror_v = 999999*ones(nA,nk,nnn);

for i_a = 1:nA
    a = Agrid(i_a);
	lna_cheby = lnAchebygrid(i_a);
    for i_k = 1:nk
        k = Kgrid(i_k);
        for i_n = 1:nnn
            n = Ngrid(i_n);
			tot_stuff = a*k^aalpha*n^(1-aalpha)+(1-ddelta)*k+z*(1-n);
			ustuff = xxi*(1-n)^(1-eeta);
			state = [log(a),log(k),log(n),tot_stuff,ustuff];
			lnk_cheby = -1 + 2*(log(k)-min_lnK)/(max_lnK-min_lnK);
			lnn_cheby = -1 + 2*(log(n)-min_lnN)/(max_lnN-min_lnN);
			lnEMH = ChebyshevND(degree,[lna_cheby,lnk_cheby,lnn_cheby])*coeff_lnmh;
			lnEMF = ChebyshevND(degree,[lna_cheby,lnk_cheby,lnn_cheby])*coeff_lnmf;
			c = 1/(bbeta*exp(lnEMH));
			q = kkappa/c/(bbeta*exp(lnEMF));
			v = (q/ustuff)^(1/(eeta-1));
			kplus = tot_stuff - c - kkappa*v;
			nplus = (1-x)*n + q*v;
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
			for i_node = 1:nA
				lnaplus = lnAgrid(i_node);
				lnaplus_cheby = -1 + 2*(lnaplus-min_lnA)/(max_lnA-min_lnA);
				if (lnaplus_cheby < -1 || lnaplus_cheby > 1)
					error('Aplus out of bound')
				end
				lnEMH_plus = ChebyshevND(degree,[lnaplus_cheby,lnkplus_cheby,lnnplus_cheby])*coeff_lnmh;
				lnEMF_plus = ChebyshevND(degree,[lnaplus_cheby,lnkplus_cheby,lnnplus_cheby])*coeff_lnmf;
				cplus = 1/(bbeta*exp(lnEMH_plus));
				qplus = kkappa/cplus/(bbeta*exp(lnEMF_plus));
				tthetaplus = (qplus/xxi)^(1/(eeta-1));
				EMH_hat = EMH_hat + P(i_a,i_node)*((1-ddelta+aalpha*exp(lnaplus)*(kplus/nplus)^(aalpha-1))/cplus);
				EMF_hat = EMF_hat + P(i_a,i_node)*(( (1-ttau)*((1-aalpha)*exp(lnaplus)*(kplus/nplus)^aalpha-z-ggamma*cplus) + (1-x)*kkappa/qplus - ttau*kkappa*tthetaplus )/cplus );
			end        

			c_imp = 1/(bbeta*EMH_hat);
			q_imp = kkappa/c_imp/(bbeta*EMF_hat);
			ttheta_imp = (q_imp/xxi)^(1/(eeta-1));
			v_imp = ttheta_imp*(1-n);

            EEerror_c(i_a,i_k,i_n) = abs((c-c_imp)/c_imp);   
            EEerror_v(i_a,i_k,i_n) = abs((v-v_imp)/v_imp);  
        end
    end
end
EEerror_c_inf = norm(EEerror_c(:),inf)
EEerror_v_inf = norm(EEerror_v(:),inf)

EEerror_c_mean = mean(EEerror_c(:));
EEerror_v_mean = mean(EEerror_v(:));

figure
plot(Kgrid,squeeze(EEerror_c(ceil(nA/2),:,ceil(nnn/2))))

figure
plot(Kgrid,squeeze(EEerror_v(ceil(nA/2),:,ceil(nnn/2))))

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
            state(1) = lnAgrid(i_A); A = state(1);
            state(2) = lnKgrid(i_k); k = state(2);
            state(3) = lnNgrid(i_n); n = state(3);
            EM = exp([1 log(state)]*[coeff_mh coeff_mf]);
            
            y = A*(k)^(aalpha)*(n)^(1-aalpha);
            c = (bbeta*EM(1))^(-1);
            ttheta = (kkappa/(c*xxi*bbeta*EM(2)))^(1/(eeta-1));
            v = ttheta*(1-n);
            mh = (1-ddelta+aalpha*y/k)/c;
            mf = ( (1-ttau)*((1-aalpha)*y/n-z-ggamma*c) + (1-x)*kkappa/xxi*ttheta^(1-eeta) - ttau*kkappa*ttheta )/c;
            w = ttau*A*k^(aalpha)*(1-aalpha)*n^(-aalpha) + (1-ttau)*(z+ggamma*c) + ttau*kkappa*ttheta;
    
            kk(i_A,i_k,i_n) = y - c +(1-ddelta)*k - kkappa*v + z*(1-nn(i_A,i_k,i_n));
            nn(i_A,i_k,i_n) = (1-x)*n + xxi*ttheta^(eeta)*(1-n);
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

%% Ergodic set where art thou?
    figure
    scatter3(Asim,ksim,nsim)
    xlabel('Productivity')
    ylabel('Capital')
    zlabel('Employment')

%% Dynamics
Aindex = ceil(nA/2);
figure
[Kmesh,Nmesh] = meshgrid(lnKgrid,lnNgrid);
DK = squeeze(kk(Aindex,:,:))-Kmesh';
DN = squeeze(nn(Aindex,:,:))-Nmesh';
quiver(Kmesh',Nmesh',DK,DN,2);
axis tight

%% Paths 1
T = 5000; scale = 0;
A = 0.6;
k1 = zeros(1,T); n1 = zeros(1,T);
k1(1) = 1100; n1(1) = 0.90;
for t = 1:T
    state = [A k1(t) n1(t)];
    EM = exp([1 log(state)]*[coeff_mh coeff_mf]);
    y = A*(k1(t))^(aalpha)*(n1(t))^(1-aalpha);
    c = (bbeta*EM(1))^(-1);
    ttheta = (kkappa/(c*xxi*bbeta*EM(2)))^(1/(eeta-1));
    v = ttheta*(1-n1(t));
    
    if t < T
    k1(t+1) = y - c +(1-ddelta)*k1(t) - kkappa*v;
    n1(t+1) = (1-x)*n1(t) + xxi*ttheta^(eeta)*(1-n1(t));
    end
end
xx = k1; y = n1;
u = [k1(2:end)-k1(1:end-1) 0];
v = [n1(2:end)-n1(1:end-1) 0];

figure
quiver(xx,y,u,v,scale,'Linewidth',0.3);



wage_export = wage_export(:);
ttheta_export = ttheta_export(:);
cc = cc(:);
kk = kk(:);
nn = nn(:);
dlmwrite('../CUDA_VFI/wage_export.csv',wage_export,'precision',16);
dlmwrite('../CUDA_VFI/ttheta_export.csv',ttheta_export,'precision',16);
dlmwrite('../CUDA_VFI/cPEA_export.csv',cc,'precision',16);
dlmwrite('../CUDA_VFI/kPEA_export.csv',kk,'precision',16);
dlmwrite('../CUDA_VFI/nPEA_export.csv',nn,'precision',16);


save('PEA_Em.mat');
