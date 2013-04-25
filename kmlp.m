function [ output, grad ] = kmlp(M, H1, H2, K, XL, XR, weights, training, ti)
%this function compute the output and gradient of the mlp
% intputs :
% M : length of the input (576 in the project)
% H1, H2 : size of the first and second layers
% XL, XR : left and right input (column vectors)
% weights : the set of weights of the MLP (column vector form)
% training : boolean, true if we currently train the mlp. if false, it will
%            not compute the backward pass and take account of ti
% ti : the class of the output. For training mode only
% outputs :
% output : =a3
% grad : the gradient induced by this sample (column vector form)

% the detail of the calculus is not explained there, but the whole 
% derivation is explained in mlp_implementation.pdf

% put the weights on a matrix form 
[W1L, B1L, W1R, B1R, W2L, B2L, W2LR, B2LR, W2R, B2R, W3, B3] = weightsToMatrix(M, H1, H2, K, weights);

%forward pass
%compute first layer + hidden part
A1L=W1L*XL + B1L;
A1R=W1R*XR +B1R;
Z1L=tanh(A1L);
Z1R=tanh(A1R);

% compute second layer + hidden part
A2L=W2L*Z1L+B2L;
A2R=W2R*Z1R+B2R;
A2LR=W2LR*[Z1L;Z1R]+B2LR;
Z2=A2LR./(1+exp(-A2L))./(1+exp(-A2R));

%compute third layer (output)
A3=W3*Z2+B3;
output=A3;

% backward pass (if training only)
if (~training)
   grad=0;
   return;
end

%third layer
r3=2*(A3-ti);
grad_W3=r3*Z2';
grad_B3=r3;

%second layer
gp2L=Z2.*(1-1./(1+exp(-A2L)));
gp2LR=(1./(1+exp(-A2L))).*(1./(1+exp(-A2R)));
gp2R=Z2.*(1-1./(1+exp(-A2R)));

r2L =diag(gp2L) * W3' * r3;
r2LR=diag(gp2LR)* W3' * r3;
r2R =diag(gp2R) * W3' * r3;

grad_W2L=r2L*(Z1L)';
grad_W2R=r2R*(Z1R)';
grad_W2LR=r2LR*([Z1L;Z1R]');

grad_B2L=r2L;
grad_B2R=r2R;
grad_B2LR=r2LR;

%first layer
gp1L=1-Z1L.^2;
gp1R=1-Z1R.^2;

r1L=diag(gp1L)*W2L'*r2L + diag(gp1L)*W2LR(:,1:H1)'*r2LR;
r1R=diag(gp1R)*W2R'*r2R + diag(gp1R)*W2LR(:,H1+1:end)'*r2LR;

grad_W1L=r1L*(XL)';
grad_W1R=r1R*(XR)';
grad_B1L=r1L;
grad_B1R=r1R;

% put the gradient induced by this instance in the form of a column vector
grad=weightsToVector(M,H1,H2, K, grad_W1L, grad_B1L, grad_W1R, grad_B1R, grad_W2L, grad_B2L, grad_W2LR, grad_B2LR, grad_W2R, grad_B2R, grad_W3, grad_B3); 
end

