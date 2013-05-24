% In this file, we implement linear multiway classification
%% load file and create dataset
load(['datasets' filesep 'norb_5class']);
% don't forget that we add a constant value
train_x=[ones(1, size(train_left_s,2));double(train_left_s);double(train_right_s)];
[train_x, mu, sigma]=normalize(train_x);
train_x(1,:)=ones(1, size(train_left_s,2));
train_t=zeros(5,size(train_cat_s,2));
for i=1:size(train_cat_s,2)
   train_t(double(train_cat_s(i))+1,i)=1; 
end
test_x=[ones(1, size(test_left_s,2));double(test_left_s); double(test_right_s)];
sigma(1)=1;
mu(1)=0;
test_x=(test_x-repmat(mu,1,size(test_x,2)))./repmat(sigma,1,size(test_x,2));
test_cat=double(test_cat_s)+1;

%% Using a squarred error
% There exists a global solution. No need for cross - validation or so,
% the evaluation will be done on the test set. 
% In fact we find 5 vector w, one for every coordinate of the output
% I took notations of course
Phi=train_x';
W_squared_error=((Phi'*Phi)\Phi'*(train_t'))';

% %% using squared error and tichonov regularizer
% % we test n different tychonov regularizer
% possible_v=1:50:1000;
% error_with_v=zeros(1,length(possible_v));
% for k=1:length(possible_v)
%     v=possible_v(k);
%     disp(strcat('testing tichonov with v=',num2str(v)));
%     
%     % each timeusing a 10 fold cross validation
%     segments=floor(linspace(1,size(train_x,2),11));
%     error=0;
%     for i=1:10
%         % at first extract the sub train set and the validation set
%         validation_segment=segments(i):segments(i+1);
%         if (i==1)
%             train_segment=(segments(1)+1):(size(train_x,2));
%         end
%         if (i==10)
%             train_segment=1:(segments(10)-1);
%         end
%         if (i~=10 && i~=1)
%            train_segment=[1:segments(i)-1,segments(i+1)+1:size(train_x,2)]; 
%         end
%         sub_train_x=train_x(:,train_segment);
%         sub_train_t=train_t(:,train_segment);
%         valid_train_x=train_x(:,validation_segment);
%         valid_train_t=train_t(:,validation_segment);
%         
%         % then find the linear classifiers
%         Phi=sub_train_x';
%         W_tichonov=((Phi'*Phi+v*eye(size(Phi,2)))\Phi'*(sub_train_t'))';
%         
%         % evaluate error on validation set
%         t_tichonov=W_tichonov*valid_train_x;
%         error_with_v(k)=error_with_v(k)+sum(sum((t_tichonov-valid_train_t).^2));
%     end
%     error_with_v(k)=error_with_v(k)/10;
%     xlabel('v')
%     ylabel('error, given by 10 folds cross validation')
%     title('evolution of error on validation set with tichnov regularizer v')
% end
% 
% % then we obtain the tichonov regularizer for which the results on 
% % cross validation are the best 
% [~,indice_optimal]=min(error_with_v);
% optimal_v=possible_v(indice_optimal);
%plot(possible_v,error_with_v);
optimal_v=400;

disp(strcat('the optimal v found is ',num2str(optimal_v)));

Phi=train_x';
W_tichonov=((Phi'*Phi+optimal_v*eye(size(Phi,2)))\Phi'*(train_t'))';

%% then we try with a logistic regression (gradient descent)
% we define arbitrarily a validation set of 1/3 of the datas
perm=randperm(size(train_t,2));
perm_train_t=train_t(:,perm);
perm_train_x=train_x(:,perm);
cesure=floor(size(train_t,2)/3);
valid_train_x=perm_train_x(:,1:cesure);
valid_train_t=perm_train_t(:,1:cesure);
sub_train_x=perm_train_x(:, (cesure+1):end);
sub_train_t=perm_train_t(:, (cesure+1):end);
% we will make a gradient descent, we initialize our W coefficient with
% random values
for z=1:20
W_logistic=randn(5,size(sub_train_x,1))/sqrt(size(sub_train_x,1));
converged=false;
k=2000;
previous_error_on_validationset=Inf;
count=0;
averaged_validation_error=0;

    while(~converged)
        count=count+1;
        disp(strcat('epock ',num2str(count)));
       for i=1:size(sub_train_x,2)
            %k=k+1;
            sigma_k=exp(W_logistic*sub_train_x(:,i));
            sigma_k=sigma_k/sum(sigma_k);
            grad_v=(sigma_k-sub_train_t(:,i))*sub_train_x(:,i)';
            W_logistic=W_logistic-1/k*grad_v;
       end

       % now we compute the error on the validation set 
        lsexp_y=log(sum(exp(W_logistic*valid_train_x),1));
        errors=lsexp_y-sum(valid_train_t.*(W_logistic*valid_train_x));

        if(sum(errors)>previous_error_on_validationset*0.95)
            converged=true;
        else
            previous_error_on_validationset=mean(errors);
        end
    end
    averaged_validation_error=averaged_validation_error+previous_error_on_validationset;
end
averaged_validation_error

%% Then we test the obtained solutions on the test set
count_squared_error=0;
count_tichonov=0;
count_logistic=0;
for i=1:size(test_x,2)
    % for squared error
    t_squared_error=W_squared_error*test_x(:,i);
    [~, class_squared_error]=max(t_squared_error);
    if (class_squared_error~=test_cat(i))
        count_squared_error=count_squared_error+1;
    end
    
    % for squared error with tichonov regularizer
    t_tichonov=W_tichonov*test_x(:,i);
    [~, class_tichonov]=max(t_tichonov);
    if (class_tichonov~=test_cat(i))
        count_tichonov=count_tichonov+1;
    end
    
    % for logistic error with tichonov regularizer
    t_logistic=W_logistic*test_x(:,i);
    [~, class_logistic]=max(t_logistic);
    if (class_logistic~=test_cat(i))
        count_logistic=count_logistic+1;
    end
end
disp('error with linear classifiers');
disp(strcat('with squared eror function ',num2str(count_squared_error), ... 
    ' misclassified items on ',num2str(size(test_cat,2))));
disp(strcat('with tichonov regularizer ',num2str(count_tichonov), ... 
    ' misclassified items on ',num2str(size(test_cat,2))));
disp(strcat('with logistic regularizer ',num2str(count_logistic), ... 
    ' misclassified items on ',num2str(size(test_cat,2))));