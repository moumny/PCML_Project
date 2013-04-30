%% compute a multi class training
load(['datasets' filesep 'norb_5class']);
train_cat_s=double(train_cat_s); 
train_left_s=double(train_left_s);
train_right_s=double(train_right_s);

M=576;
H1=10;
H2=10;
K=5;
momentum=0;
minibatch=1;

[optimal_weights, error]=trainMultiMLP(M, H1, H2, K, train_left_s,train_right_s, train_cat_s,momentum);

plot(error);