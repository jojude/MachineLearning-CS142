function [pass, maxErr] = grad_check(fun, sizeVec, rep, tol)
    if nargin < 4
        tol = exp(-6);
    end
    
    maxErr = 0;
    for i = 1 : rep
        wO = randn(sizeVec(1), sizeVec(2));
        [g1] = gradest(fun, wO);
        [f, g2] = fun(wO);
        maxErr = max(maxErr, max(abs(g1(:) - g2(:))));
    end
    
    if maxErr < tol
        pass = 1;
    else
        pass = 0;
    end
end