function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];

y_m = zeros(m,num_labels); %Initiate Y matrix

for i = 1:size(y_m,1);
  v = zeros(1,num_labels); %Initiate replacement vector
  v(y(i)) = 1;
  y_m(i,:) = v; %Matrix 5000x10
end

for u = 1:m
  a_2 = sigmoid(X(u,:)*Theta1');
  a_2 = [ones(1,1) a_2]; %1x26
  hx = sigmoid(a_2*Theta2'); %1x10
  J = J + (-y_m(u,:)*log(hx)' - (1-y_m(u,:))*log(1-hx)');
end

J = J/m;

The1_sq = Theta1(:,2:size(Theta1,2));
The1_sq = The1_sq.*The1_sq;
The2_sq = Theta2(:,2:size(Theta2,2));
The2_sq = The2_sq.*The2_sq;

J = J + lambda/(2*m) * (sum(sum(The1_sq))+sum(sum(The2_sq)));


a_1 = X'; %401x5000
z_2 = Theta1*a_1; %25x5000
a_2 = sigmoid(z_2); %25x5000
a_2 = [a_2;ones(1,m)]; %26x5000
z_3 = Theta2*a_2; %10x5000
a_3 = sigmoid(z_3); %10x5000
d_3 = a_3 - y_m'; %10x5000
d_2 = Theta2(:,2:end)' * d_3 .* sigmoidGradient(z_2); %25x5000
Theta1_grad = d_2*a_1'; %d2*a1 is 25x401 this is D1
Theta2_grad = d_3*a_2'; %d3*a2 is 10x26 this is D2

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
