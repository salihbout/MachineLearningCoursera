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

%PART 1
%Preparing useful variables
X = [ones(m, 1) X];

Y = zeros(m,num_labels);

%Converting the labels number to vector of 0 and 1
for i = 1:m
   
  Y(i,y(i)) = 1;
  
end


%Computing the final activation value : feedforward
  a2 = sigmoid( X*Theta1');

  a2 = [ones(size(a2,1), 1) a2];

  g = sigmoid(a2*Theta2');
  
  
%Computing the cost function : J

  J = (1/m)*sum( sum((-log(g).*Y) - (log(1-g).*(1-Y))) ) ;


%Regularization term


regulTerm = (lambda/(2*m))*(sum( sum(Theta1(:,2:size(Theta1,2)).^2)) + sum(sum(Theta2(:,2:size(Theta2,2)).^2)));


%regularized cost function
J = J + regulTerm;


%PART 2


%Preparing Variables



a1 = X;
z2 = X*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];
z3= a2*Theta2';
a3 = sigmoid(z3);

%Computing errors
  
  deltha3 = a3 - Y;

  deltha2 = (deltha3*Theta2) .*sigmoidGradient( [ones(size(z2,1),1) z2] );
  
%Computing accumulated gradients

  deltha2NoFirst = deltha2(:,2:end);
  
  DELTHA1 =  deltha2NoFirst'*a1;
  DELTHA2 =  deltha3'*a2;

  

  Theta1_grad = DELTHA1./m;
  Theta2_grad = DELTHA2./m;
  


%Adding Regularization terms

  Theta1Bais = Theta1;
  Theta2Bais = Theta2;
  
  for i = 1:size(Theta1Bais,1),
    Theta1Bais(i,1) = 0;
  end
  
  for i = 1:size(Theta2Bais,1),
    Theta2Bais(i,1) = 0;
  end
  
  
  
  Theta1_grad += (lambda/m)*Theta1Bais;
  Theta2_grad += (lambda/m)*Theta2Bais;

  















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
