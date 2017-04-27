function [f,G] = cross_entropy(X,w,y)    
    y_hat = sigmoid(X * w);
    
    f = (-y' * log(y_hat))- ((1-y') * log(1 - y_hat));
    G = X' * (y_hat - y);
end
