function w = minLoo(X,y)
    f = [0;0;1];
    A = [ X, -ones(size(X,1),1) ; -X, -ones(size(X,1),1)];
    b = [ y ; -y];
    
    z = linprog(f,A,b);
    w = z(1:end-1);
    delta = z(end);
end