function [lambda, b] = dual_softmargin(K,y, beta)

 H = diag(y)*K*diag(y)*(1/beta);
 Aeq = y';
 beq = 0;
 A = [eye(size(y,1),size(y,1)); -eye(size(y,1),size(y,1))];
 b = [ones(size(y,1),1); zeros(size(y,1),1)];
 f = -ones(size(y,1),1);
 
 lambda = quadprog(H,f,A,b,Aeq,beq);
 
 idx = (0.01 <= lambda) & (lambda <= (1 - 0.01));
 b = mean(y(idx) - ((1/beta)*(K(idx,:)*diag(y)*lambda)));
 
end