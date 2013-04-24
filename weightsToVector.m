function [ weights ] = weightsToVector( M, H1, H2, K, W1L, B1L, W1R, B1R, W2L, B2L, W2LR, B2LR, W2R, B2R, W3, B3 )
% this function transform the set of weights in one single vector
% so we will manipulate easily the set of weights during the 
% gradient descent
% note that the total length of this vector is : 
total_length= 2*H1*(M+1) + 4*H2*H1+3*H2 + (H2+1)*K;

weights=zeros(total_length,1);
start_point=1;

%%%%%%%%%%%%%
%first layer
%%%%%%%%%%%%%
weights(start_point:start_point-1+M*H1)=reshape(W1L,M*H1,1);
start_point=start_point+M*H1;
weights(start_point:start_point-1+H1)=reshape(B1L,H1,1);
start_point=start_point+H1;

weights(start_point:start_point-1+M*H1)=reshape(W1R,M*H1,1);
start_point=start_point+M*H1;
weights(start_point:start_point-1+H1)=reshape(B1R,H1,1);
start_point=start_point+H1;

%%%%%%%%%%%%%
%second layer
%%%%%%%%%%%%%

weights(start_point:start_point-1+H1*H2)=reshape(W2L,H1*H2,1);
start_point=start_point+H1*H2;
weights(start_point:start_point-1+H2)=reshape(B2L,H2,1);
start_point=start_point+H2;

weights(start_point:start_point-1+2*H1*H2)=reshape(W2LR,2*H1*H2,1);
start_point=start_point+2*H1*H2;
weights(start_point:start_point-1+H2)=reshape(B2LR,H2,1);
start_point=start_point+H2;

weights(start_point:start_point-1+H1*H2)=reshape(W2R,H1*H2,1);
start_point=start_point+H1*H2;
weights(start_point:start_point-1+H2)=reshape(B2R,H2,1);
start_point=start_point+H2;


%%%%%%%%%%%%%
%third layer
%%%%%%%%%%%%%

weights(start_point:start_point-1+H2*K)=reshape(W3,H2,K);
start_point=start_point+H2*K;
weights(start_point:start_point-1+K)=B3;
start_point=start_point+K;

if start_point-1~=total_length
    disp('weightsToVector\: there must be an error here');
end
end

