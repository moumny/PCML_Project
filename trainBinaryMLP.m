function [ optimal_weights, logisticError, mu_and_sigmas, misclass, missclass_vector, training_error] = trainBinaryMLP( M, H1, H2, left_inputs, right_inputs, labels, learning_rate,momentum )
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
% logistic_error : the error on the validation set through time
% mu_and_sigmas=[mu_left, sigma_left, mu_right, sigma_right]

weights=initializeWeights(M,H1,H2, 1);
weights_1=zeros(length(weights),1);

misclass=0;

% check if the labels are valid
if (find(labels~=1 & labels~=-1))
    disp('error in training of the binary mlp. The labels vectors must be composed of -1 and 1');
    return;
end
targets=labels;

%Split the train set and normalize
[left_train,right_train,cat_train,left_valid,right_valid,cat_valid] = splitTrainSet(left_inputs, right_inputs, targets); 
[left_train_norm, mu_left, sigma_left]  = normalize(left_train);
[right_train_norm, mu_right, sigma_right] = normalize(right_train);
left_valid_norm = normalize(left_valid, mu_left, sigma_left);
right_valid_norm = normalize(right_valid, mu_right, sigma_right);

% return mus and sigma in order to use it on the test set, later
mu_and_sigmas=[mu_left, sigma_left, mu_right, sigma_right];
missclass_vector=[];
training_error=[];
logisticError = zeros(1,20);
early_stopping = false;
epoch=1;

%learning rate
lr= learning_rate;

while early_stopping == false  & epoch < 21
    %train
    for i=randperm(size(left_train_norm,2))
        [~,gradient]=mlp(M,H1,H2,left_train_norm(:,i),right_train_norm(:,i),weights,true,cat_train(i));
        weights_new=weights - lr*(1-momentum)*gradient + momentum*(weights-weights_1);
        weights_1=weights;
        weights=weights_new;
    end
    %lr=lr/1.2;
    %Compute error
    error = 0;
    num = zeros(1,size(left_valid_norm,2));
    for i=1:size(left_valid_norm,2)
        [output,~]=mlp(M,H1,H2,left_valid_norm(:,i),right_valid_norm(:,i),weights,false,0);
        error = error + accuLog(-cat_valid(i)*output); 
        num(i)=cat_valid(i)*output;
    end
    size(find(num>0))
    
    logisticError(epoch)= error/size(left_valid,2);
    missclass_vector=[missclass_vector, size(find(num>0),2)];
    
    % made for report : compute error on training set 
        %Compute error on validation set 
    error = 0;
    num = zeros(1,size(left_train_norm,2));
    for i=1:size(left_train_norm,2)
        [output,~]=mlp(M,H1,H2,left_train_norm(:,i),right_train_norm(:,i),weights,false,0);
        error = error + accuLog(-cat_train(i)*output); 
        num(i)=cat_train(i)*output;
    end
    training_error = [training_error (error/size(left_train,2))];


%     if epoch > 10
%         early_stopping = logisticError(end) >= logisticError(end-1);
%     end
    
    epoch = epoch + 1
end


optimal_weights=weights;

end

