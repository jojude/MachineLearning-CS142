function [xvals, fvals] = gradient_decent(func, xO, options)
    [f,G] = func(xO);
    G = G';
    xvals = xO';
    fvals = [];
    fvals(1) = f;
    
    alpha = options.StepSize();
    
    for i = 1:options.NumIterations()
        new_xval = xvals(i,:) - alpha * G(i,:);
        xvals = [xvals; new_xval];
        
        [new_f,new_G] = func(new_xval');
        new_G = new_G';
        fvals(i+1) = new_f;
        G = [G; new_G];
    end    
end