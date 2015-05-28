%% Parametric Policy Function Iteration a la Benitez-Silva, Hall, Hitsch, and Rust 2000
clc;
clear;
close all;
addpath(genpath('tools'));
addpath(genpath('../PEA'))
tic


%% Accuracy Control
mypara;
minA = 0.3; maxA = 1.7;
minK = 500; maxK = 3000;
minN = 0.5; maxN = 0.999;
degree = 5;
tol = 1e-7*(1-bbeta);
damp = 0.5;
nA = 16;
nK = 16;
nN = 16;
maxiter = 1;

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
Agrid = ChebyshevRoots(nA,'Tn',[0.95,1.05]);
Kgrid = ChebyshevRoots(nK,'Tn',[1300,1400]);
Ngrid = ChebyshevRoots(nN,'Tn',[0.93,0.95]);
Achebygrid = ChebyshevRoots(nA,'Tn');
Kchebygrid = ChebyshevRoots(nK,'Tn');
Nchebygrid = ChebyshevRoots(nN,'Tn');
[fakebasis,order_table] = ChebyshevND(degree,[0,0,0]);
N = nA*nK*nN; K = size(order_table,1);
P = zeros(N,K);
for i = 1:N
    [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
	for i_term = 1:K
		orders = order_table(i_term,:);
		P(i,i_term) = chebypoly(orders(1),Achebygrid(i_a))*chebypoly(orders(2),Kchebygrid(i_k))*chebypoly(orders(3),Nchebygrid(i_n));
	end
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
pphi = zeros(K,1); % coefficients of value function w.r.t basis
[epsi_nodes,weight_nodes] = GH_nice(6,0,1);
n_nodes = length(epsi_nodes);
policy = zeros(N,2); exitflag = zeros(N,1); util = zeros(N,1);
options = optimoptions('fmincon',...
                       'Algorithm','interior-point',...
					   'AlwaysHonorConstraints','bounds',...
					   'Display','notify-detailed',...
					   'MaxFunEvals',5000,...
					   'TolFun',1e-3,...
					   'MaxIter',5000,...
					   'DerivativeCheck','off',...
					   'GradObj','off',...
					   'Diagnostics','off',...
					   'TypicalX',[c_ss,(1-n_ss)]);

%% Use some guess
parfor i = 1:N
    [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
	policy(i,:) = [c_ss, (1-n_ss)];
end

%% Main body of iteration
value_diff = 10;
iter = 0;
while ((value_diff > tol) && (iter < maxiter))
    %% Given policy find value function
    EP = zeros(N,K);
    for i = 1:N
        [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
        a = Agrid(i_a); k = Kgrid(i_k); n = Ngrid(i_n);
        
        % GH to find EP given policy function
		c = policy(i,1); v = policy(i,2);
		util(i) = log(c) - ggamma*n; % compute utility by the way
        kplus = tot_stuff(i) - c - kkappa*v;
	   	nplus = (1-x)*n + ustuff(i)*v^eeta;
        kplus_cheby = -1 + 2*(kplus-minK)/(maxK-minK);
        nplus_cheby = -1 + 2*(nplus-minN)/(maxN-minN);
        
		for i_term = 1:K
			orders = order_table(i_term,:);
			temp_EP = 0;
       		for i_node = 1:n_nodes
				eps = epsi_nodes(i_node);
				aplus = exp(rrho*log(a) + ssigma*eps);
				aplus_cheby = -1 + 2*(aplus-minA)/(maxA-minA);
				if ((aplus_cheby > 1) || (aplus_cheby < -1))
					disp(a);
					disp(aplus);
				end
				temp_EP = temp_EP + weight_nodes(i_node)*chebypoly(orders(1),aplus_cheby)*chebypoly(orders(2),kplus_cheby)*chebypoly(orders(3),nplus_cheby);
			end
			EP(i,i_term) = temp_EP;
		end
	end
    
    X = (P - bbeta*EP);
    pphi_temp = (X'*X)\(X'*util);
    pphi_new = (1-damp)*pphi_temp + damp*pphi; 

    %% Find diff
    value_diff = norm(P*(pphi_new-pphi),Inf)
    pphi = pphi_new;
	
    %% Given value find policy function
    parfor i = 1:N
        [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
        a = Agrid(i_a); k  = Kgrid(i_k); n = Ngrid(i_n); %#ok<PFBNS>
		max_v = ((maxN-(1-x)*n)/ustuff(i))^(1/eeta);
        lb = [1e-5,1e-7]; ub = [tot_stuff(i)-minK-kkappa*max_v,max_v-1e-5];
        state = [a,k,n,tot_stuff(i),ustuff(i)];
		x0 = policy(i,:);
        [policy(i,:),exitflag(i)] = nested_obj(state,param,pphi,epsi_nodes,weight_nodes,n_nodes,x0,lb,ub,options);
		str = sprintf('State is %f, %f, %f, %f, %f, policy is %f, %f',[state,policy(i,:)]);
		disp(str);
	end

	iter = iter + 1;

end

%% End game
tic
save
