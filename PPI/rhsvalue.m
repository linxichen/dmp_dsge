function value = rhsvalue(state,control,param,pphi,epsi_nodes,weight_nodes,n_nodes)
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
kplus_cheby = -1 + 2*(kplus-minK)/(maxK-minK);
nplus_cheby = -1 + 2*(nplus-minN)/(maxN-minN);

% Find current utility
v = ((nplus - (1-x)*n)/ustuff)^(1/eeta); % v is guaranteed to be positive outside
c = tot_stuff - kplus - kkappa*v;
if (c < 0)
    value = -9e10;
    return;
else
    util = log(c) - ggamma*n;
end

% Find expected value
EV = 0;
for i_node = 1:n_nodes
    eps = epsi_nodes(i_node);
    aplus = exp(rrho*log(a) + ssigma*eps);
    aplus_cheby = -1 + 2*(aplus-minA)/(maxA-minA);
    if ((aplus_cheby > 1) || (aplus_cheby < -1))
        a
        aplus
    end
    temp_V = ChebyshevND(degree,[aplus_cheby kplus_cheby nplus_cheby])*pphi;
    EV = EV + weight_nodes(i_node)*temp_V;
end

% return rhs value
value = util + bbeta*EV;

end