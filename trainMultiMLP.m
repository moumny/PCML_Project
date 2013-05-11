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

%parameters for early stoping
early_stopping = false;
sliding_window=5;
mean_on_W_epocks=Inf;
last_good_weight=0;


epoch=1;

%learning rate
lr=0.01;

while ( early_stopping == false  && epoch < 50 )
    %train
    for i=randperm(size(left_train_norm,2))
        [~,gradient]=kmlp(M,H1,H2, K, left_train_norm(:,i),right_train_norm(:,i),weights,true,cat_train(:,i));
        weights_new=weights - lr*(1-momentum)*gradient + momentum*(weights-weights_1);
        weights_1=weights;
        weights=weights_new;
       % lr=lr+1;
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
    validationError = [validationError (error/size(left_valid,2))];
    disp(strcat('epoch : ',num2str(epoch),',~ ',num2str(size(find(num>0),2)),' are correctly classified on the validation set (total=',num2str(length(num)),')'));

    if (mod(epoch,sliding_window)==0)
        % if the mean on 5 epocks seems higher than last time
        mean_on_W_epocks_new=mean(validationError(end-sliding_window+1:end));
        disp(strcat('average error on the last ',num2str(sliding_window),' epocks : ',num2str(mean_on_W_epocks_new)));
        if (mean_on_W_epocks_new>0.8*mean_on_W_epocks)
            early_stopping=true;
            weights=last_good_weight;
            disp('early stopping');
        else
            last_good_weight=weights;
            mean_on_W_epocks=mean_on_W_epocks_new;
        end
    end
    epoch = epoch + 1;
end
optimal_weights=weights;

end

