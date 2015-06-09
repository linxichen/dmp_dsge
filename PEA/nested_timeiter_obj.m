function [x,fval,exitflag] = nested_timeiter_obj(state,param,coeff_c,coeff_v,lnAgrid,lnAchebygrid,P,nA,x0,options)
% Call fmincon
[x,fval,exitflag] = fsolve(@eulers,x0,options);

function [residual] = eulers(control)
% load parameters
bbeta = param(1); % 1
ggamma = param(2); % 2
kkappa = param(3); % 3
eeta = param(4); % 4
rrho = param(5); %5
ssigma = param(6); %6
min_lnA = param(7);  %7
max_lnA = param(8); %8
min_lnK = param(9); %9
max_lnK = param(10); %10
min_lnN = param(11); % 11
max_lnN = param(12); % 12
degree = param(13); % 13
x = param(14); % 14
aalpha = param(15);
ddelta = param(16);
xxi = param(17);
ttau = param(18);
z = param(19);

% Load variables
lna = state(1); lnk = state(2); lnn = state(3); tot_stuff = state(4); ustuff = state(5); i_a = state(6);
c = control(1);
v = control(2);
q = ustuff*v^(eeta-1);
kplus = tot_stuff - c - kkappa*v;
nplus = (1-x)*exp(lnn) + ustuff*v^(eeta);
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
    lnaplus_cheby = lnAchebygrid(i_node);
    if (lnaplus_cheby < -1 || lnaplus_cheby > 1)
        error('Aplus out of bound') 
    end
    cplus = ChebyshevND(degree,[lnaplus_cheby,lnkplus_cheby,lnnplus_cheby])*coeff_c;
    vplus = ChebyshevND(degree,[lnaplus_cheby,lnkplus_cheby,lnnplus_cheby])*coeff_v;
    qplus = xxi*(1-nplus)^(1-eeta)*vplus^(eeta-1);
    tthetaplus = vplus/(1-nplus);
    EMH_hat = EMH_hat + P(i_a,i_node)*((1-ddelta+aalpha*exp(lnaplus)*(kplus/nplus)^(aalpha-1))/cplus);
    EMF_hat = EMF_hat + P(i_a,i_node)*(( (1-ttau)*((1-aalpha)*exp(lnaplus)*(kplus/nplus)^aalpha-z-ggamma*cplus) + (1-x)*kkappa/qplus - ttau*kkappa*tthetaplus )/cplus );
end

% Find violation in Euler equations
residual(1) = 1/c - bbeta*EMH_hat;
residual(2) = kkappa/c/q - bbeta*EMF_hat;

end

end
