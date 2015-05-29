function [x,fval,exitflag] = nested_obj(state,param,pphi,epsi_nodes,weight_nodes,n_nodes,x0,lb,ub,options);
% Call fmincon
[x,fval,exitflag] = fmincon(@obj,x0,[],[],[],[],lb,ub,@pos_constraint,options);

function [c,ceq] = pos_constraint(control)
% This function calculates implied consumption and vacancy value such that they are positive
 ggamma = param(2); % 2
 kkappa = param(3); % 3
 eeta = param(4); % 4
 x = param(14);

a = state(1); k = state(2); n = state(3); tot_stuff = state(4); ustuff = state(5);
kplus = control(1);
nplus = control(2);
vacancy = ((nplus - (1-x)*n)/ustuff)^(1/eeta);
consumption = tot_stuff - kplus - kkappa*vacancy;
c(1) = 1e-3 - consumption;
c(2) = -vacancy;
ceq = 0;

end

function [value,gradient] = obj(control)
	[value,gradient] = rhsvalue(control);
	value = -value;
	gradient = -gradient;
end

function [value,gradient] = rhsvalue(control)
gradient = zeros(2,1);

% load parameters
bbeta = param(1); % 1
ggamma = param(2); % 2
kkappa = param(3); % 3
eeta = param(4); % 4
rrho = param(5); %5
ssigma = param(6); %6
minA = param(7);  %7
maxA = param(8); %8
minK = param(9); %9
maxK = param(10); %10
minN = param(11); % 11
maxN = param(12); % 12
degree = param(13); % 13
x = param(14); % 14

% Load variables
a = state(1); k = state(2); n = state(3); tot_stuff = state(4); ustuff = state(5);
kplus = control(1);
nplus = control(2);
log_kplus = log(kplus);
log_nplus = log(nplus);

% Find current utility
v = ((nplus - (1-x)*n)/ustuff)^(1/eeta); % v is guaranteed to be positive outside
c = tot_stuff - kplus - kkappa*v;
if (c < 0)
	value = -9e10;
	gradient = -9e10*ones(2,1);
	return;
else
	util = log(c) - ggamma*n;
	Du(1) = 1/c*(-1);
	Du(2) = 1/c*(-kkappa)/eeta*v^(1-eeta);
end

% Find expected value
EV = 0;
DEV_k = 0;
DEV_n = 0;
for i_node = 1:n_nodes
	eps = epsi_nodes(i_node);
	log_aplus = rrho*log(a) + ssigma*eps;
	log_state = [1 log_aplus log_kplus log_nplus];
	temp_V = exp(log_state*pphi);
	EV = EV + weight_nodes(i_node)*temp_V;

	% Compute DEV at aplus
	DEV_k = DEV_k + weight_nodes(i_node)*temp_V*pphi(3)/kplus;
	DEV_n = DEV_n + weight_nodes(i_node)*temp_V*pphi(4)/nplus;
end

% return rhs value
value = util + bbeta*EV;
gradient(1) = Du(1) + bbeta*DEV_k;
gradient(2) = Du(2) + bbeta*DEV_n;

end

end
