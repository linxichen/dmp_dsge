%% Parametric Policy Function Iteration a la Benitez-Silva, Hall, Hitsch, and Rust 2000
clc;
clear;
close all;
addpath(genpath('tools'));
addpath(genpath('../PEA'))
tic


%% Accuracy Control
mypara;
minA = 0.4; maxA = 1.6;
minK = 1000; maxK = 2000;
minN = 0.5; maxN = 0.999;
degree = 3;
tol = 1e-3;
damp = 0.2;
nA = 11;
nK = 11;
nN = 11;

%% Encapsulate all parameters
param = [... 
 bbeta; % 1
 ggamma; % 2
 kkappa; % 3
 eeta; % 4
 rrho; %5
 ssigma; %6
 minA;  %7
 maxA; %8
 minK; %9
 maxK; %10
 minN; % 11
 maxN; % 12
 degree; % 13
 x; % 14
 ];
%% Grid generation
% Agrid = ChebyshevRoots(degree,'Tn',[minA,maxA]);
Agrid = ChebyshevRoots(nA,'Tn',[0.7,1.3]);
Kgrid = ChebyshevRoots(nK,'Tn',[1100,1600]);
Ngrid = ChebyshevRoots(nN,'Tn',[0.85,0.99]);
Achebygrid = ChebyshevRoots(nA,'Tn');
Kchebygrid = ChebyshevRoots(nK,'Tn');
Nchebygrid = ChebyshevRoots(nN,'Tn');
[fakebasis,order_table] = ChebyshevND(degree,[0,0,0]);
N = nA*nK*nN; K = size(order_table,1);
P = zeros(N,K);
parfor i = 1:N
    [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
    P(i,:) = ChebyshevND(degree,[Achebygrid(i_a),Kchebygrid(i_k),Nchebygrid(i_n)]); %#ok<PFBNS>
end

%% Precomputation
tot_stuff = zeros(N,1); ustuff = zeros(N,1);
parfor i = 1:N
    [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
    a = Agrid(i_a); k  = Kgrid(i_k); n = Ngrid(i_n); %#ok<PFBNS>
    tot_stuff(i) = a*k^aalpha*n^(1-aalpha) + (1-ddelta)*k + z*(1-n);
    ustuff(i) = xxi*(1-n)^(1-eeta);
end


%% Initialize policy function and value functions
kopt = k_ss*ones(N,1);
kopt_cheby = -1 + (kopt-minK)/(maxK-minK)*2;
nopt = n_ss*ones(N,1);
nopt_cheby = -1 + (nopt-minN)/(maxN-minN)*2;
pphi = zeros(K,1); % coefficients of value function w.r.t basis
[n_nodes,epsi_nodes,weight_nodes] = GH_Quadrature(10,1,1);
policy = zeros(N,2); exitflag = zeros(N,1); util = zeros(N,1);
options = optimoptions('fmincon','Algorithm','sqp','AlwaysHonorConstraints','bounds','Display','notify-detailed');

value_diff = 10;
while value_diff > tol
    %% Given value find policy function
    parfor i = 1:N
        [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
        a = Agrid(i_a); k  = Kgrid(i_k); n = Ngrid(i_n); %#ok<PFBNS>
        lb = [minK,(1-x)*n]; ub = [maxK,maxN];
        state = [a,k,n,tot_stuff(i),ustuff(i)];
        objfunc = @(x) -rhsvalue(state,x,param,pphi,epsi_nodes,weight_nodes,n_nodes);
        [policy(i,:),~,exitflag(i)] = fmincon(objfunc,[k,0.95*((1-x)*n)+0.05*(1-(1-x)*n)],[],[],[],[],lb,ub,[],options);
    end
    
    %% Given policy find value function
    EP = zeros(N,K);
    parfor i = 1:N
        [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
        a = Agrid(i_a);
        
        % GH to find EP given policy function
        kplus = policy(i,1); nplus = policy(i,2);
        kplus_cheby = -1 + 2*(kplus-minK)/(maxK-minK);
        nplus_cheby = -1 + 2*(nplus-minN)/(maxN-minN);
        
        for i_node = 1:n_nodes
            eps = epsi_nodes(i_node);
            aplus = exp(rrho*log(a) + ssigma*eps);
            aplus_cheby = -1 + 2*(aplus-minA)/(maxA-minA);
            if ((aplus_cheby > 1) || (aplus_cheby < -1))
                a
                aplus
            end
            temp_EP = ChebyshevND(degree,[aplus_cheby kplus_cheby nplus_cheby]);
            EP(i,:) = EP(i,:) + weight_nodes(i_node)*temp_EP;
        end
    end
    
    %% Find utility and regress
    parfor i = 1:N
        [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
        a = Agrid(i_a); k = Kgrid(i_k); n = Ngrid(i_n);
        kplus = policy(i,1);
        nplus = policy(i,2);
        v = ((nplus - (1-x)*n)/ustuff(i))^(1/eeta); % v is guaranteed to be positive outside
        c = tot_stuff(i) - kplus - kkappa*v;
        util(i) = log(c) - ggamma*n;
    end
    X = (P - bbeta*EP);
    pphi_temp = (X'*X)\(X'*util);
    pphi_new = (1-damp)*pphi_temp + damp*pphi; 
    
    %% Find diff
    value_diff = norm(pphi_new-pphi,inf)
    pphi = pphi_new;
    
end

%% End game
tic
save
