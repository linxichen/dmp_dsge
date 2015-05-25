load PEA.mat
meanA = mean(Asim); meank = mean(ksim); meann = mean(nsim);
ssigmaA = std(Asim); ssigmak = std(ksim); ssigman = std(nsim);
rhs = [((Asim-meanA)/ssigmaA)' ((ksim-meank)/ssigmak)' ((nsim-meann)/ssigman)'];
X = hermite_tensor(rhs(1:end-1,:),3);
bbeta_mf = (X'*X)\(X'*mfsim(2:end)');
bbeta_mh = (X'*X)\(X'*mhsim(2:end)');
save('initial_6degree.mat','bbeta_mf','bbeta_mh');
save('scale.mat','meanA','meank','meann','ssigmaA','ssigmak','ssigman');
