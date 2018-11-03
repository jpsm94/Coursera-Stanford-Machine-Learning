function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
num_theta = length(theta);    % number of theta parameters
tmp_theta = zeros(num_theta); % temporary

J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
	% non-vectorized
    %for j = 1:num_theta
    %  tmp_theta(j) = theta(j) - alpha/m * sum((X * theta - y) .* X(:, j));
    %end
    %for j = 1:num_theta % simultaneous update of theta
    %  theta(j) = tmp_theta(j)
    %end
	
	% vectorized
	  theta = theta - alpha/m .* (X' * (X * theta - y));
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    % fprintf('Gradient Descent iteration=%i, cost=%f\n', iter, J_history(iter));
end

end
