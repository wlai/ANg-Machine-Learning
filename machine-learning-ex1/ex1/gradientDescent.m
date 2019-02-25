function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    size_theta = length(theta);
    next_theta = zeros(1,size_theta);
    for theta_index = 1:size_theta
        Cost_i = 0;
        for i = 1:m
            V = X(i,:);
            h_theta = dot(theta', V);
            Cost_i = Cost_i + ((h_theta - y(i)) * X(i,theta_index));
        end
        next_theta(theta_index) = theta(theta_index) - (alpha * Cost_i / m);
        % simultaneous update of theta
    end
    
    for theta_index = 1:size_theta
        theta(theta_index) = next_theta(theta_index);
    end

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
