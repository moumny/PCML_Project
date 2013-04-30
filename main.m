%% compute a binary training
load(['datasets' filesep 'norb_binary_truckcar']);
train_cat_s=double(2*train_cat_s-7); % change 1 and 3 to -1 and 1
train_left_s=double(train_left_s);
train_right_s=double(train_right_s);

M=576;
H1=4;
H2=8;
momentum=0;
minibatch=1;

[optimal_weights, error]=trainBinaryMLP(M, H1, H2, train_left_s,train_right_s, train_cat_s,momentum);

plot(error);