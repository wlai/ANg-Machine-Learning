function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
% 

% Computed cost function J, unregularized
J = sum((((X * Theta') - Y).^2) .* R,'all') / 2;

% Adjust for regularization
J = J + (lambda / 2) * (sum(Theta.^2,'all') + sum(X.^2, 'all')) ;


% Compute gradients
%{
% Functional but slow: doing it in loops
for k = 1:num_features 
    % i is the index to users
    % j is the index to movies
    % k is the index to features
    for i = 1:num_movies 
        for j = 1:num_users
            if R(i,j) == 1
                temp = ((X(i,:) * Theta (j,:)') - Y(i,j));
                X_grad(i,k) =  X_grad(i,k) + (temp * Theta(j,k));
                Theta_grad(j,k) =  Theta_grad(j,k) + (temp * X(i,k));
            end
        end  
    end
end
%}        


for i=1:num_movies
    % find all non-zero j's/col's on this i-th row of R, 
    % i.e. all users who have rated this i-th movie
    idx = find(R(i,:)== 1);  
    
    % temp matrices
    Theta_temp = Theta(idx,:); % find thetas for the users (rows) in idx
    Y_temp = Y(i,idx); % find the idx-users' score for i-th movie
    
    % Gradient of X(i)
    X_grad(i,:) = ((X(i,:) * Theta_temp') - Y_temp) * Theta_temp;
    
    % Adjust for Gradient
    X_grad(i,:) = X_grad(i,:) + lambda * X(i,:);
end

for j=1:num_users
    % find all non-zero i's/rows's on this j-th column of R, 
    % i.e. all movies which have been rated by this j-th user
    idx = find(R(:,j)== 1); 
    
    % temp matrices
    X_temp = X(idx,:);
    Y_temp = Y(idx,j);
    
    % Gradient of Theta(j)
    Theta_grad(j,:) = (((X_temp * Theta(j,:)') - Y_temp)' * X_temp);  
    
    % Adjust for Gradient
    Theta_grad(j,:) = Theta_grad(j,:) + lambda * Theta(j,:);
end















% =============================================================


grad = [X_grad(:); Theta_grad(:)];

end
