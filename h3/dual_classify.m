function [yhat] = dual_classify(Ktest, lambda, b, y, beta)
    yhat = sign((Ktest * diag(y) * lambda * (1/beta)) + b);
end
