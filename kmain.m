%% compute a multi class training
load(['datasets' filesep 'norb_5class']);
train_cat_s=double(train_cat_s); 
train_left_s=double(train_left_s);
train_right_s=double(train_right_s);

M=576;
K=5;
H1=8;
H2=2;
momentum=0.5;
learning_rate=0.01;

[optimal_weights, error, mu_and_sigmas]=trainMultiMLP(M, H1, H2, K, train_left_s,train_right_s, train_cat_s,learning_rate,momentum);

plot(error);

%% test the classifier obtained with the optimal weights on the test set
test_left_norm=normalize(double(test_left_s), mu_and_sigmas(:,1), mu_and_sigmas(:,2));
test_right_norm=normalize(double(test_right_s), mu_and_sigmas(:,3), mu_and_sigmas(:,4));

count_error=0;
for i=1:size(test_left_norm,2)
    output=kmlp(M,H1,H2,5,test_left_norm(:,i),test_right_norm(:,i),optimal_weights, false);
    [~,indice_max]=max(output);
    if (indice_max~=(test_cat_s(i)+1))
        count_error=count_error+1;
    end
end

count_error