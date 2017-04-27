function w = minL1(X,y)
    f = [0;0;ones(size(X,1),1);];
    A = [ X, -(eye(size(X,1))); -X, -(eye(size(X,1)))];
    b = [ y; -y];
    
    z = linprog(f,A,b);
    w = z(1:2);
    delta = z(3:end);
end