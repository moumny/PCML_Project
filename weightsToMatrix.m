function [W1L, B1L, W1R, B1R, W2L, B2L, W2LR, B2LR, W2R, B2R, W3, B3] = weightsToMatrix(M, H1, H2, weights)
% this function takes in intput a vector of weights of the right size and 
% express them in the form of matrices
start_point=1;

%%%%%%%%%%%%%%
%first layer
%%%%%%%%%%%%%%

W1L=reshape(weights(start_point:start_point+M*H1-1),H1,M);
start_point=start_point+M*H1;
B1L=reshape(weights(start_point:start_point+1*H1-1),H1,1);
start_point=start_point+1*H1;

W1R=reshape(weights(start_point:start_point+M*H1-1),H1,M);
start_point=start_point+M*H1;
B1R=reshape(weights(start_point:start_point+1*H1-1),H1,1);
start_point=start_point+1*H1;

%%%%%%%%%%%%%%%%%
%second layer
%%%%%%%%%%%%%%%%%
W2L=reshape(weights(start_point:start_point+H2*H1-1),H2,H1);
start_point=start_point+H2*H1;
B2L=reshape(weights(start_point:start_point+1*H2-1),H2,1);
start_point=start_point+1*H2;

W2LR=reshape(weights(start_point:start_point+2*H2*H1-1),H2,2*H1);
start_point=start_point+2*H2*H1;
B2LR=reshape(weights(start_point:start_point+1*H2-1),H2,1);
start_point=start_point+1*H2;

W2R=reshape(weights(start_point:start_point+H2*H1-1),H2,H1);
start_point=start_point+H2*H1;
B2R=reshape(weights(start_point:start_point+1*H2-1),H2,1);
start_point=start_point+1*H2;

%%%%%%%%%%%%%%%%%%%%
%third layer
%%%%%%%%%%%%%%%%%%%%

W3=reshape(weights(start_point:start_point+H2-1),1,H2);
start_point=start_point+H2;

B3=weights(start_point);
if start_point~=length(weights)
    disp('there must be an error here');
end

end

