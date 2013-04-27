function [ optimal_weights, validationError ] = trainMultiMLP( M, H1, H2, K, left_inputs, right_inputs, labels, momentum )
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

weights=initializeWeights(M,H1,H2,K);
weights_1=zeros(length(weights),1);

% Create target values 
targets = zeros(K,size(labels,2)); %maybe -1 instead of 0 ? no. maybe normalize?
for i=1:size(labels)
    targets(labels(i)+1,i)=1;
end

 
%Split the train set and normalize
[left_train,right_train,cat_train,left_valid,right_valid,cat_valid] = splitTrainSet(left_inputs, right_inputs, targets); 
left_train_norm = normalize(left_train);
right_train_norm = normalize(right_train);
left_valid_norm = normalize(left_valid);
right_valid_norm = normalize(right_valid);

validationError = [];
early_stopping = false;
epoch=1;
while early_stopping == false  & epoch < 100 
    %train
    for i=randperm(size(left_train_norm,2))
        [~,gradient]=kmlp(M,H1,H2, K, left_train_norm(:,i),right_train_norm(:,i),weights,true,cat_train(i));
        weights_new=weights - 1/i*(1-momentum)*gradient + momentum*(weights-weights_1);
        weights_1=weights;
        weights=weights_new;
    end
    
    %Compute error
    error = 0;
    num = zeros(1,size(left_valid_norm,2));
    for i=1:size(left_valid_norm,2)
        [output,~]=kmlp(M,H1,H2,K, left_valid_norm(:,i),right_valid_norm(:,i),weights,false,0);
        error = error + norm(output-cat_valid(i)); 
%         num(i)=cat_valid(i)*output;
    end
%     size(find(num>0))
    
    validationError = [validationError (error/size(left_valid,2))];

    if epoch > 10
        early_stopping = validationError(end) > validationError(end-1);
    end
    
    epoch = epoch + 1
end
optimal_weights=weights;

end

