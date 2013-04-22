function [ optimal_weights, logisticError ] = trainBinaryMLP( M, H1, H2, left_inputs, right_inputs, labels, momentum )
% This function train the mlp for binary dataset
% inputs : 
% M : size of inputs
% H1 : size first layer
% H2 : size second layer
% left_inputs, right_input : inputs of mlp. each of size (M x
% NumberOfInputs)
% labels : vector of ti. !!! must composed of 1 and -1
%  Momentum : the momentum for gradient descent
% output : 
% optimal_weights : well, guess...

weights=initializeWeights(M,H1,H2);
weights_1=zeros(length(weights),1);

% check if the labels are valid
if (find(labels~=1 & labels~=-1))
    disp('error in training of the binary mlp. The labels vectors must be composed of -1 and 1');
    return;
end

[left_train,right_train,cat_train,left_valid,right_valid,cat_valid] = splitTrainSet(left_inputs, right_inputs, labels); 
left_train_norm = normalize(left_train);
right_train_norm = normalize(right_train);
left_valid_norm = normalize(left_valid);
right_valid_norm = normalize(right_valid);

logisticError = [];
early_stopping = false;
epoch=1;
while early_stopping == false  & epoch < 100 
    %train
    for i=randperm(size(left_train_norm,2))
        [~,gradient]=mlp(M,H1,H2,left_train_norm(:,i),right_train_norm(:,i),weights,true,cat_train(i));
        weights_new=weights - 1/i*(1-momentum)*gradient + momentum*(weights-weights_1);
        weights_1=weights;
        weights=weights_new;
    end
    
    %Compute error
    error = 0;
    num = zeros(1,size(left_valid_norm,2));
    for i=1:size(left_valid_norm,2)
        [output,~]=mlp(M,H1,H2,left_valid_norm(:,i),right_valid_norm(:,i),weights,false,0);
        error = error + accuLog(-cat_valid(i)*output); 
        num(i)=cat_valid(i)*output;
    end
    size(find(num>0))
    
    logisticError = [logisticError (error/size(left_valid,2))];

    if epoch > 2
        early_stopping = logisticError(end) >= logisticError(end-1);
    end
    
    epoch = epoch + 1
end
optimal_weights=weights;

end

