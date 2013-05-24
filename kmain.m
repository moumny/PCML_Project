%% compute a multi class training
load(['datasets' filesep 'norb_5class']);
train_cat_s=double(train_cat_s); 
train_left_s=double(train_left_s);
train_right_s=double(train_right_s);

% % this part is for the additional dataset 
% load(['facult_dataset' filesep 'training_cat']);
% train_cat_s=double(training_cat);
% load(['facult_dataset' filesep 'training_set_left']);
% train_left_s=double(training_set_left);
% load(['facult_dataset' filesep 'training_set_right']);
% train_right_s=double(training_set_right);
% 
% load(['facult_dataset' filesep 'testing_cat']);
% test_cat_s=double(testing_cat);
% load(['facult_dataset' filesep 'testing_set_left']);
% test_left_s=double(testing_set_left);
% load(['facult_dataset' filesep 'testing_set_right']);
% test_right_s=double(testing_set_right);

%%


M=576;
K=5;
H1=60;
H2=40;
momentum=0;
learning_rate=0.01;

master_count_error=zeros(10,1);
average_confmat=zeros(5);

%for z=1:15
z=1;
[optimal_weights, error, mu_and_sigmas, missclass, misclassvector, training_error]=trainMultiMLP(M, H1, H2, K, train_left_s,train_right_s, train_cat_s,learning_rate,momentum);
[haxes,hline1,hline2] =plotyy(1:length(error),[training_error; error],1:length(error), 1800-misclassvector);
axes(haxes(1))
ylabel('Errors on validation and training set')
axes(haxes(2))
ylabel('elements')
legend('correctly classified on val. set','error on training set','error on validation set');



%% test the classifier obtained with the optimal weights on the test set
test_left_norm=normalize(double(test_left_s), mu_and_sigmas(:,1), mu_and_sigmas(:,2));
test_right_norm=normalize(double(test_right_s), mu_and_sigmas(:,3), mu_and_sigmas(:,4));

count_error=0;
confusion_matrix=zeros(5);
for i=1:size(test_left_norm,2)
    output=kmlp(M,H1,H2,5,test_left_norm(:,i),test_right_norm(:,i),optimal_weights, false);
    [~,indice_max]=max(output);
    confusion_matrix(test_cat_s(i)+1,indice_max)= confusion_matrix(test_cat_s(i)+1,indice_max)+1;
    if (indice_max~=(test_cat_s(i)+1))
        count_error=count_error+1;
    end
end
master_count_error(z)=count_error;
average_confmat=average_confmat+confusion_matrix;
%end
mean(master_count_error)
std(master_count_error)
average_confmat