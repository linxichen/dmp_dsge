%% Parametric Policy Function Iteration a la Benitez-Silva, Hall, Hitsch, and Rust 2000
clc;
clear;
close all;
addpath(genpath('../tools'));
addpath(genpath('../PEA'))
tic


%% Accuracy Control
mypara;
minA = 0.3; maxA = 1.7;
minK = 700; maxK = 2000;
minN = 0.5; maxN = 0.99;
degree = 5;
tol = (1-bbeta);
damp = 0.0;
nA = 16;
nK = 50;
nN = 50;

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
Agrid = ChebyshevRoots(nA,'Tn',[0.8,1.2]);
Kgrid = ChebyshevRoots(nK,'Tn',[800,2000]);
Ngrid = ChebyshevRoots(nN,'Tn',[0.8,0.99]);
Achebygrid = ChebyshevRoots(nA,'Tn');
Kchebygrid = ChebyshevRoots(nK,'Tn');
Nchebygrid = ChebyshevRoots(nN,'Tn');
N = nA*nK*nN;

%% Precomputation
X = ones(N,4);
tot_stuff = zeros(N,1); ustuff = zeros(N,1);
parfor i = 1:N
    [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
    a = Agrid(i_a); k  = Kgrid(i_k); n = Ngrid(i_n); %#ok<PFBNS>
	X(i,:) = [1,log(a),log(k),log(n)]
    tot_stuff(i) = a*k^aalpha*n^(1-aalpha) + (1-ddelta)*k + z*(1-n);
    ustuff(i) = xxi*(1-n)^(1-eeta);
end


%% Initialize policy function and value functions
pphi = [7.255147667181310;0.010568982383841;0.036625376586577;-0.002025003070139]; % coefficients of value function w.r.t basis
[epsi_nodes,weight_nodes] = GH_nice(21,0,1);
n_nodes = length(epsi_nodes);
policy = zeros(N,2); exitflag = zeros(N,1); util = zeros(N,1);
v_new = zeros(N,1);

% Nonlinear regression options
opts = statset('nlinfit');
opts.Display = 'final';
opts.MaxIter = 1000;

options = optimoptions('fmincon',...
                       'Algorithm','sqp',...
					   'AlwaysHonorConstraints','bounds',...
					   'Display','notify-detailed',...
					   'MaxFunEvals',3000,...
					   'TolFun',1e-6,...
					   'MaxIter',3000,...
					   'DerivativeCheck','off',...
					   'GradObj','on',...
					   'Diagnostics','off',...
					   'TypicalX',[k_ss,n_ss]);

%% Use some guess
parfor i = 1:N
    [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
	policy(i,:) = [max(minK,(1-ddelta)*Kgrid(i_k)),max(minN,(1-x)*Ngrid(i_n))];
end

%% Main body of iteration
value_diff = 10;
while value_diff > tol
    %% Given value find policy function
    parfor i = 1:N
        [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
        a = Agrid(i_a); k  = Kgrid(i_k); n = Ngrid(i_n); %#ok<PFBNS>
        lb = [500,(1-x)*n]; ub = [3000,0.999];
        state = [a,k,n,tot_stuff(i),ustuff(i)];
		x0 = [k,n];
        [policy(i,:),temp,exitflag(i)] = nested_obj(state,param,pphi,epsi_nodes,weight_nodes,n_nodes,x0,lb,ub,options);
		v_new(i) = -temp;
		str = sprintf('State is %f, %f, %f, %f, %f, policy is %f, %f',[state,policy(i,:)]);
		disp(str);
    end

    %% Given policy find value function
    % EP = zeros(N,K);
    % parfor i = 1:N
    %     [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
    %     a = Agrid(i_a);
    %     
    %     % GH to find EP given policy function
    %     kplus = policy(i,1); nplus = policy(i,2);
    %     
    %     for i_node = 1:n_nodes
    %         eps = epsi_nodes(i_node);
    %         aplus = exp(rrho*log(a) + ssigma*eps);
    %         if ((aplus_cheby > 1) || (aplus_cheby < -1))
    %             a
    %             aplus
    %         end
    %         temp_EP = ChebyshevND(degree,[aplus_cheby kplus_cheby nplus_cheby]);
    %         EP(i,:) = EP(i,:) + weight_nodes(i_node)*temp_EP;
    %     end
    % end
    % 
    % %% Find utility and regress
    % parfor i = 1:N
    %     [i_a,i_k,i_n] = ind2sub([nA,nK,nN],i);
    %     n = Ngrid(i_n);
    %     kplus = policy(i,1);
    %     nplus = policy(i,2);
    %     v = ((nplus - (1-x)*n)/ustuff(i))^(1/eeta); % v is guaranteed to be positive outside
    %     c = tot_stuff(i) - kplus - kkappa*v;
    %     util(i) = log(c) - ggamma*n;
    % end
    % X = (P - bbeta*EP);
    % pphi_temp = (X'*X)\(X'*util);
    % pphi_new = (1-damp)*pphi_temp + damp*pphi; 

	%% Regress to get new pphi
	regeqn = @(b,x) exp(b(1)+b(2).*(x(:,2))+b(3).*(x(:,3))+b(4).*(x(:,4))); 
	pphi_temp = nlinfit(X,v_new,regeqn,pphi,opts);
    pphi_new = (1-damp)*pphi_temp + damp*pphi; 

    %% Find diff
    value_diff = norm((pphi_new-pphi),Inf);
    pphi = pphi_new;
	

end

%% End game
tic
save

%% Euler equation error
nK_ee = 10; nA_ee = 10; nN_ee = 10;
Kee = linspace(0.8*k_ss,1.2*k_ss,nK_ee);
Aee = linspace(0.8,1.2,nA_ee);
Nee = linspace(0.7,0.999,nN_ee);
EEerror_c = 999999*ones(nA_ee*nK_ee*nN_ee,1);
EEerror_v = 999999*ones(nA_ee*nK_ee*nN_ee,1);
cc = zeros(nA_ee*nK_ee*nN_ee,1);

for index = 1:nA_ee*nK_ee*nN_ee
    [i_a,i_k,i_n] = ind2sub([nA_ee,nK_ee,nN_ee],index);
    a = Aee(i_a);
    k = Kee(i_k);
    n = Nee(i_n);
    state = [a,k,n,a*k^aalpha*n^(1-aalpha) + (1-ddelta)*k + z*(1-n),xxi*(1-n)^(1-eeta)];
    
    x0 = [k,n];
    lb = [500,(1-x)*n]; ub = [3000,0.999];
    [temp_policy,~,~] = nested_obj(state,param,pphi,epsi_nodes,weight_nodes,n_nodes,x0,lb,ub,options);
    kplus = temp_policy(1); nplus = temp_policy(2);
    v = ((nplus - (1-x)*n)/state(5))^(1/eeta);
    c = state(4) - kplus - kkappa*v;
    cc(index) = c;
    
    % Find expected mf and mh and implied consumption
    Emf = 0; Emh = 0;
    for i_node = 1:length(weight_nodes)
        Aplus = exp(rrho_A*log(a) + ssigma_A*epsi_nodes(i_node));
        stateplus = [Aplus,kplus,nplus,Aplus*kplus^aalpha*nplus^(1-aalpha) + (1-ddelta)*kplus + z*(1-nplus),xxi*(1-nplus)^(1-eeta)];

        x0 = [kplus,nplus];
        lb = [500,(1-x)*nplus]; ub = [3000,0.999];
        [temp_policy,~,~] = nested_obj(stateplus,param,pphi,epsi_nodes,weight_nodes,n_nodes,x0,lb,ub,options);
        kplusplus = temp_policy(1); nplusplus = temp_policy(2);
        vplus = ((nplusplus - (1-x)*nplus)/stateplus(5))^(1/eeta);
        cplus = stateplus(4) - kplusplus - kkappa*vplus;
        tthetaplus = vplus/(1-nplus); yplus = Aplus*kplus^aalpha*nplus^(1-aalpha);
        
        Emh = Emh + weight_nodes(i_node)*((1-ddelta+aalpha*yplus/kplus)/cplus);
        Emf = Emf + weight_nodes(i_node)*(( (1-ttau)*((1-aalpha)*yplus/nplus-z-ggamma*cplus) + (1-x)*kkappa/xxi*tthetaplus^(1-eeta) - ttau*kkappa*tthetaplus )/cplus );
    end
    c_imp = (bbeta*Emh)^(-1);
    v_imp = (kkappa/c_imp/(bbeta*Emf)/state(5))^(1/(eeta-1));
    EEerror_c(index) = abs((c-c_imp)/c_imp);
    EEerror_v(index) = abs((v-v_imp)/v_imp);
end

EEerror_c_inf = norm(EEerror_c(:),inf)
EEerror_v_inf = norm(EEerror_v(:),inf)
