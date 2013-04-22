%% compute a binary training
load(['datasets' filesep 'norb_binary']);
train_cat_s=double(train_cat_s-2); % change 1 and 3 to -1 and 1
train_left_s=double(train_left_s);
train_right_s=double(train_right_s);

M=576;
H1=3;
H2=2;
momentum=0;
minibatch=1;

[optimal_weights, error]=trainBinaryMLP(M, H1, H2, train_left_s,train_right_s, train_cat_s,momentum);

plot(error);