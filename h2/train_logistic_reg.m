function [wOpt, Objval] = train_logistic_reg(X,y,b)
    obj_func = @(w) obj(X, w, y, b);
    [t,n] = size(X);
    wO = zeros(n, 1);
    options = optimoptions('fminunc','Algorithm', 'quasi-newton', 'GradObj', 'on');
    
    [wOpt, Objval, exitflag, output] = fminunc(obj_func, wO, options)
end
