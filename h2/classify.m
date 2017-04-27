function [yhat, phat] = classify(X,w)
    sig = sigmoid(X*w);
    phat = sig .*100;
    
    [t,n] = size(sig);
    yhat = zeros(t,1);
    
    for i = 1:t
        if sig(i) > 0.5
            yhat(i) = 1;
        else
            yhat(i) = 0;
        end
    end
    
end