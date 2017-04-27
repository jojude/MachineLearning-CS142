function [K] = poly_kernel(X1, X2, d)
    X = X1*X2';
    K = (1 + X/100).^d;
    % c = 1/100 , without the constant the dual error with a
    % poly kernel will give us 100% error for d > 1. 
end