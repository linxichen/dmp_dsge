%% Parametric Policy Function Iteration a la Benitez-Silva, Hall, Hitsch, and Rust 2000
clc;
clear;
close all;
addpath(genpath('tools'));
addpath(genpath('../PEA'))




%% Accuracy Control
mypara;
minA = 0.1; maxA = 1.9;
minK = 0.5*k_ss; maxK = 1.5*k_ss;
minN = 0.1; maxN = 0.999;
degree = 10;

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
Agrid = linspace(0.75,1.25,11);
Kgrid = ChebyshevRoots(degree,'Tn',[minK,maxK]);
Ngrid = ChebyshevRoots(degree,'Tn',[minN,maxN]);
Achebygrid = ChebyshevRoots(degree,'Tn');
Kchebygrid = ChebyshevRoots(degree,'Tn');
Nchebygrid = ChebyshevRoots(degree,'Tn');
[fakebasis,order_table] = ChebyshevND(degree,[0,0,0]);
N = degree^3; K = size(order_table,1);
P = zeros(N,K);
parfor i = 1:N
    [i_a,i_k,i_n] = ind2sub([degree,degree,degree],i);
    P(i,:) = ChebyshevND(degree,[Achebygrid(i_a),Kchebygrid(i_k),Nchebygrid(i_n)]); %#ok<PFBNS>
end

%% Initialize policy function and value functions
kopt = k_ss*ones(N,1);
kopt_cheby = -1 + (kopt-minK)/(maxK-minK)*2;
nopt = n_ss*ones(N,1);
nopt_cheby = -1 + (nopt-minN)/(maxN-minN)*2;
pphi = zeros(K,1); % coefficients of value function w.r.t basis
[n_nodes,epsi_nodes,weight_nodes] = GH_Quadrature(7,1,1);
controls = zeros(N,2); exitflag = zeros(N,1);
options = optimoptions('fmincon','Algorithm','sqp','AlwaysHonorConstraints','bounds','Display','notify-detailed');

%% Given value find policy function
parfor i = 1:N
    [i_a,i_k,i_n] = ind2sub([degree,degree,degree],i);
    a = Agrid(i_a); k  = Kgrid(i_k); n = Ngrid(i_n); %#ok<PFBNS>
    tot_stuff = a*k^aalpha*n^(1-aalpha) + (1-ddelta)*k + z*(1-n);
    ustuff = xxi*(1-n)^(1-eeta);
    lb = [minK,(1-x)*n]; ub = [maxK,maxN];
    state = [a,k,n,tot_stuff,ustuff];
    objfunc = @(x) -rhsvalue(state,x,param,pphi,epsi_nodes,weight_nodes,n_nodes);
    [controls(i,:),~,exitflag(i)] = fmincon(objfunc,[k,0.95*((1-x)*n)+0.05*(1-(1-x)*n)],[],[],[],[],lb,ub,[],options);
end


%%
save
