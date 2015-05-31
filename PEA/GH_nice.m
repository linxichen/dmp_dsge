function [x,w] = GH_nice(n,mmu,variance)
% User friendly verison of Gauss-Hermite integration.
% Formula: Den Haan, to Find E[h(y)], where y ~ N(mmu,ssigma^2), you compute
% for i = 1:n
%    E[h(y)] = E[h(y)] + w*h(x(i))
% end
% where w = w^GH/sqrt(pi), and x(i) = sqrt(2)*ssigma*xxi(i)^GH + mmu, w^GH
% and xxi(i)^GH are "original" Gauss-Hermite weights and nodes. So what I
% do just make things easier for you.
[x,w] = GaussHermite(n);
x = sqrt(2)*sqrt(variance).*x + mmu;
w = w/sqrt(pi);
end