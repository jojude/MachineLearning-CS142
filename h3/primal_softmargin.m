function [w, b] = primal_softmargin(X, y, beta)
 sizeX = size(X);
 A1 = [zeros(sizeX(1),1) , zeros(sizeX), -eye(sizeX(1), sizeX(1))];
 b1 = zeros(sizeX(1),1);
 
 j = (-diag(y))*[ ones(sizeX(1), 1), X];
 A2 = [j, -eye(sizeX(1), sizeX(1))];
 b2 = -ones(sizeX(1), 1);
 
 A = [A2; A1];
 b = [b2; b1];
 f = [zeros(sizeX(2)+1, 1); ones(sizeX(1),1)];
 H = [zeros(1), zeros(1,sizeX(2)),zeros(1,sizeX(1));
     zeros(sizeX(2),1), beta*eye(sizeX(2)), zeros(sizeX(2),sizeX(1));
     zeros(sizeX(1),1), zeros(sizeX), zeros(sizeX(1), sizeX(1))];
 
 w = quadprog(H,f,A,b);
 
 b = w(1);
 w = w(2:sizeX(2)+1);
end