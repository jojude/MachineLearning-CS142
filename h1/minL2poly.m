function c = minL2poly(x,y,d)
    X = [];
    for i = 1:size(x,1)
        v =[];
        for j = 1:d+1
            v = [v ; x(i)^(d-j+1)];
        end
        X = [X , v];
    end
    X = X';
    c = (X' * X) \ (X' * y);
end