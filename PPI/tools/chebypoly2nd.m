function val = chebypoly2nd(p,x)
% input: 
%   p = order of chebyshev polynomial of the second kind
%   x = argument of function
% output:
%   val = function value
% Author: Linxi Chen

%% Basic Error Checking
if (x>1 || x<-1)
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
        val = 2.*x;
    case 2
        val = 4*x.*x - 1;
    case 3
        val = 8*x.*x.*x - 4*x;
    otherwise
        old = 8*x.*x.*x - 4*x;
        oldold = 4*x.*x - 1;
        for i_order = 4:p
            val = 2*x*old - oldold;
            oldold = old;
            old = val;
        end
end

end
