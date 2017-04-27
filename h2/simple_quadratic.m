function [f, G] = simple_quadratic(x)
    f = ((3*x(1) - 9)^2) + ((x(2)-4)^2);
    
    G = [18*(x(1)-3);2*(x(2)-4)];
end
