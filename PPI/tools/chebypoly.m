function val = chebypoly(p,x)
% input: 
%   p = order of chebyshev polynomial
%   x = argument of function
% output:
%   val = function value
% Author: Linxi Chen

%% Basic Error Checking
if ((x>1) || (x<-1))
    error('argument is not bounded in [-1,1]');
end

if (mod(p,1)~=0)
    error('order is not an integer.');
end

%% Computing 
switch p
    case 0
        val = 1;
    case 1
        val = x;
    case 2
        val = 2*x*x - 1;
    case 3
        val = 4*x*x*x - 3*x;
    otherwise
        old = 4*x*x*x - 3*x;
        oldold = 2*x*x - 1;
        for i_order = 4:p
            val = 2*x*old - oldold;
            oldold = old;
            old = val;
        end
end

end
