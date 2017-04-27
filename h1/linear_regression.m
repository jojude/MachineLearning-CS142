function [mean_train_errQ2, mean_test_errQ2] = hw1()
    t_train = 10; 				% training size
    t_test = 1000;				% test size

    n = 2;                                  % dimension
    u = [0; ones(n-1,1)];                   % target weights
    sigma = 0.1;                            % noise level
    
   
    %[error_train, error_test] = oneTimeTesting(t_train, t_test, n, u, sigma);
    
    %[avg_error_train, avg_error_test] = avgError(t_train, t_test, n, u, sigma); 
    [mean_train_errQ2, mean_test_errQ2] = overfitQ2();
    %[avg_mean_err,  avg_test_mean_err] = avg_overfitQ2();
    
    %m1 = X_train*u + randn(t_train,1)*sigma;
    %m2 = X_train*u + randn(t_train,1)./randn(t_train,1)*sigma;
    %m3 = X_train*u + randn(t_train,1) .* randn(t_train,1)*sigma;

    %y_test = X_test*u + randn(t_test,1)*sigma;
    %y_test = X_test*u + randn(t_test,1)./randn(t_test,1)*sigma
    %y_test = X_test*u + randn(t_test,1) .* randn(t_test,1)*sigma;

    
end

function [mean_train_errQ2, mean_test_errQ2] = overfitQ2()
    t = 10; % training size
    te = 1000; %testing size
    sigma = 0.1; % noise level
    
    x = rand(t, 1);                 % training patterns
    %y = double(x > 0.5);
    y = 0.5 - 10.4*x.*(x-0.5).*(x-1)+sigma*randn(t, 1);
    
    x_test = rand(te, 1);          %testing patterns    
    %y_test = double(x_test > 0.5);
    y_test = 0.5 - 10.4*x_test.*(x_test-0.5).*(x_test-1)+sigma*randn(te, 1);
    
    c1 = minL2poly(x,y,1);
    c3 = minL2poly(x,y,3);
    c5 = minL2poly(x,y,5);
    c9 = minL2poly(x,y,9);
    
    %get the mean sum of squares error 
    mean_train_errQ2 = mean_error_Q2(c1,c3,c5,c9,x,y);
    mean_test_errQ2 = mean_error_Q2(c1,c3,c5,c9,x_test, y_test);
    
    clf
    axis([0 1 -0.5 1.5])
    hold
    plot(x', y', 'k*')
    xx = (0:1000)/1000;
    %yy = double(xx > 0.5);
    yy = 0.5 - 10.4*xx.*(xx-0.5).*(xx-1);
    plot(xx, yy, 'k:')
    plot(xx, polyval(c1, xx), 'r-')
    plot(xx, polyval(c3, xx), 'g-')
    plot(xx, polyval(c5, xx), 'b-')
    plot(xx, polyval(c9, xx), 'm-')
    print -deps experiment.1.2.<m1>.ps
     
end

function [avg_mean_err,  avg_test_mean_err] = avg_overfitQ2()
    t = 10; % training size
    te = 1000; %testing size
    sigma = 0.1; % noise level
    avg_mean_err = 0;
    avg_test_mean_err = 0;
    
    for p = 1:100
        x = rand(t, 1); % training patterns
        %y = double(x > 0.5);    
        y = 0.5 - 10.4*x.*(x-0.5).*(x-1)+sigma*randn(t, 1);
        
        c1 = minL2poly(x,y,1);
        c3 = minL2poly(x,y,3);
        c5 = minL2poly(x,y,5);
        c9 = minL2poly(x,y,9);
    
        %get the mean sum of squares error 
        avg_mean_err = avg_mean_err + mean_error_Q2(c1,c3,c5,c9,x,y);
        
        x_test = rand(te,1);
        %y_test = double(x_test > 0.5);
        y_test = 0.5 - 10.4*x_test.*(x_test-0.5).*(x_test-1)+sigma*randn(te, 1);
        avg_test_mean_err = avg_test_mean_err + mean_error_Q2(c1,c3,c5,c9,x_test,y_test);
        
    end
        avg_mean_err = avg_mean_err ./ 100;
        avg_test_mean_err = avg_test_mean_err ./ 100;
end

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

function mean_errQ2 = mean_error_Q2(c1,c3,c5,c9,x,y)

    mean_errQ2 = zeros(4,1);
    mean_errQ2(1,1) = mean((polyval(c1,x) - y).^2);
    mean_errQ2(2,1) = mean((polyval(c3,x) - y).^2);
    mean_errQ2(3,1) = mean((polyval(c5,x) - y).^2);
    mean_errQ2(4,1) = mean((polyval(c9,x) - y).^2);
    
end

function [error_train, error_test] = oneTimeTesting(t_train, t_test, n, u, sigma)
    X_train = [ones(t_train,1) rand(t_train,n-1)];              % training pattern
    y_train = X_train*u + randn(t_train,1)*sigma;
    
    X_test = [ones(t_test,1) rand(t_test,n-1)];            % test pattern
    y_test = X_test*u + randn(t_test,1)*sigma;
    
    [error_train, w1_train, w2_train, woo_train] = training(X_train, y_train);
    error_test = error_train + testing(X_test, y_test, w1_train, w2_train, woo_train);
end

function [error_train, error_test] = avgError(t_train, t_test, n, u, sigma)
    error_train = 0;
    error_test = 0;
    
    for p = 1:100
        X_train = [ones(t_train,1) rand(t_train,n-1)];         % training pattern
        y_train = X_train*u + randn(t_train,1) .* randn(t_train,1)*sigma;
        
        [err_train,w1_train, w2_train, woo_train] = training(X_train, y_train);
    
        error_train = error_train + err_train;
        
        X_test = [ones(t_test,1) rand(t_test,n-1)];            % test pattern
        y_test = X_test*u + randn(t_test,1) .* randn(t_test,1)*sigma;
    
        error_test = error_train + testing(X_test, y_test, w1_train, w2_train, woo_train);
    end
        error_train = error_train ./ 100;
        error_test = error_test ./ 100;
end

function error_test = testing(X_test, y_test, w1_test, w2_test, woo_test)   
    % hypothesis for test data
    h1_test = X_test * w1_test;
    h2_test = X_test * w2_test;
    hoo_test = X_test * woo_test;
    
    %error testing for test data
    error_test = errorTest(h1_test,h2_test,hoo_test,y_test);
end

function [err_train, w1_train, w2_train, woo_train] = training(X_train, y_train)
    w1_train = minL1(X_train,y_train);
    w2_train = minL2(X_train,y_train);
    woo_train = minLoo(X_train,y_train);
    
    %plot training data
    %plotTrainingData(X_train, y_train, w1_train, w2_train, woo_train);

    % hypothesis
    h1_train = X_train * w1_train;
    h2_train = X_train * w2_train;
    hoo_train = X_train * woo_train;
    
    %error testing
    err_train = errorTest(h1_train,h2_train,hoo_train,y_train);
end


function error = errorTest(h1,h2,hoo,y)
    error = zeros(3,3);
    %w1
    error(1,1) = errorL1(h1, y);
    error(1,2) = errorL2(h1, y);
    error(1,3) = errorLoo(h1, y);
    
    %w2
    error(2,1) = errorL1(h2, y);
    error(2,2) = errorL2(h2, y);
    error(2,3) = errorLoo(h2, y);
    
    %w3
    error(3,1) = errorL1(hoo, y);
    error(3,2) = errorL2(hoo, y);
    error(3,3) = errorLoo(hoo, y);
end

function plotTrainingData(X,y,w1,w2,woo)
    clf
    plot((X(:,2)'), y', 'k*')
    hold
    plot([0 1], [w2(1) sum(w2)], 'r-')
    plot([0 1], [w1(1) sum(w1)], 'g-')
    plot([0 1], [woo(1) sum(woo)], 'b-')
    print -deps experiment.1.1.<m>.ps
end

function w = minL2(X,y)
    w = (X' * X) \ (X' * y);
end 

function w = minLoo(X,y)
    f = [0;0;1];
    A = [ X, -ones(size(X,1),1) ; -X, -ones(size(X,1),1)];
    b = [ y ; -y];
    
    z = linprog(f,A,b);
    w = z(1:end-1);
    delta = z(end);
end

function w = minL1(X,y)
    f = [0;0;ones(size(X,1),1);];
    A = [ X, -(eye(size(X,1))) ; -X, -(eye(size(X,1)))];
    b = [ y; -y];
    
    z = linprog(f,A,b);
    w = z(1:2);
    delta = z(3:end);
end

function errL2 = errorL2(h,y)
    errL2 = sum((h - y).^2);
end

function errLoo = errorLoo(h,y)
    errLoo = max(abs(h - y));
end 

function errL1 = errorL1(h,y)
    errL1 = sum(abs(h - y));
end