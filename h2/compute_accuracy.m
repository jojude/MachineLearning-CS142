function accuracy = compute_accuracy(y, yhat)
    [t,n] = size(yhat);
    accuracy = 0;
    
    for i = 1:t
        if y(i) == yhat(i)
            accuracy = accuracy + 1;
        end
    end
    
    accuracy = accuracy / t;
end