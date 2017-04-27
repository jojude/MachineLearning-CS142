function f = sigmoid(z)
    f = (1 + exp(-z)).^(-1);
end