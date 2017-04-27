load dataA3Q1.mat

beta = 1;

% linear primal
[w,bp] = primal_softmargin(X,y,beta); 
yh = sign(X * w + bp);
linear_train_error_primal = mean(yh ~= y);
yht = sign(Xtest * w + bp);
linear_test_error_primal = mean(yht ~= ytest);

% linear dual
K = linear_kernel(X,X);
[lambda,bd] = dual_softmargin(K,y,beta);
yh = dual_classify(K, lambda, bd, y, beta);
linear_train_error_dual =  mean(yh ~= y);

Ktest = linear_kernel(Xtest, X);
yht = dual_classify(Ktest, lambda, bd, y, beta);
linear_test_error_dual =  mean(yht ~= ytest);

% poly dual
d = 2;

K = poly_kernel(X, X, d);
[lambda,bd] = dual_softmargin(K,y,beta);
yh = dual_classify(K, lambda, bd, y, beta);
poly_train_error_dual =  mean(yh ~= y);

Ktest = poly_kernel(Xtest, X, d);
yht = dual_classify(Ktest, lambda, bd, y, beta);
poly_test_error_dual =  mean(yht ~= ytest);

%gauss dual
sigma = 10;

K = gauss_kernel(X, X, sigma);
[lambda, bd] = dual_softmargin(K,y,beta);
yh = dual_classify(K, lambda, bd, y, beta);
gauss_train_error_dual = mean(yh ~= y);

Ktest = poly_kernel(Xtest, X, d);
yht = dual_classify(Ktest, lambda, bd, y, beta);
gauss_test_error_dual = mean(yht ~= ytest);

%print results 
given = fprintf('beta : %d , sigma : %d, d : %d \n\n', beta,sigma,d);

r1 = fprintf('linear_primal_train : %d \n',linear_train_error_primal);
r2 = fprintf('linear_primal_test: %d \n', linear_test_error_primal);
r3 = fprintf('linear_dual_train: %d \n', linear_train_error_dual);
r4 = fprintf('linear_dual_test: %d \n\n', linear_test_error_dual);

r5 = fprintf('ploy_dual_train: %d \n', poly_train_error_dual);
r6 = fprintf('poly_dual_test: %d \n\n', poly_test_error_dual);

r7 = fprintf('gauss_dual_train: %d \n', gauss_train_error_dual);
r8 = fprintf('gauss_dual_test: %d \n', gauss_test_error_dual);
