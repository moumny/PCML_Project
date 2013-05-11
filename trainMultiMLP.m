function [ optimal_weights, validationError, mu_and_sigmas] = trainMultiMLP( M, H1, H2, K, left_inputs, right_inputs, labels, momentum )
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
% validation_error : the error on the validation set through time
% mu_and_sigmas=[mu_left, sigma_left, mu_right, sigma_right]

weights=initializeWeights(M,H1,H2,K);
weights_1=zeros(length(weights),1);

% Create target values 
targets = zeros(K,size(labels,2)); %maybe -1 instead of 0 ? no. maybe normalize?
for i=1:size(labels,2)
    targets(labels(i)+1,i)=1;
end

 
%Split the train set and normalize
[left_train,right_train,cat_train,left_valid,right_valid,cat_valid] = splitTrainSet(left_inputs, right_inputs, targets); 
[left_train_norm, mu_left, sigma_left]  = normalize(left_train);
[right_train_norm, mu_right, sigma_right] = normalize(right_train);
left_valid_norm = normalize(left_valid, mu_left, sigma_left);
right_valid_norm = normalize(right_valid, mu_right, sigma_right);

% return mus and sigma in order to use it on the test set, later
mu_and_sigmas=[mu_left, sigma_left, mu_right, sigma_right];


validationError = [];
early_stopping = false;
epoch=1;

%learning rate
lr=0.001;

while early_stopping == false  & epoch < 100 
    %train
    for i=randperm(size(left_train_norm,2))
        [~,gradient]=kmlp(M,H1,H2, K, left_train_norm(:,i),right_train_norm(:,i),weights,true,cat_train(:,i));
        weights_new=weights - lr*(1-momentum)*gradient + momentum*(weights-weights_1);
        weights_1=weights;
        weights=weights_new;
    end
    
    %Compute error
    error = 0;
    num = zeros(1,size(left_valid_norm,2));
    for i=1:size(left_valid_norm,2)
        [output,~]=kmlp(M,H1,H2,K, left_valid_norm(:,i),right_valid_norm(:,i),weights,false,0);
        error = error +sum( (output-cat_valid(:,i)).^2); 
        % this is used to count the number of well classified point on the
        % validation set
        [~, class_max]=max(output);
        [~, true_max]=max(cat_valid(:,i));
        num(i)=-1;
        if (class_max==true_max)
            num(i)=1;
        end
    end
    size(find(num>0))
    
    validationError = [validationError (error/size(left_valid,2))];

    if epoch > 20
        early_stopping = validationError(end) > validationError(end-1);
    end
    
    epoch = epoch + 1
end
optimal_weights=weights;

end

