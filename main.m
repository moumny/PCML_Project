%% compute a binary training

% if the following is true, then we use the dataset coming from truck and
% car dataset, otherwise, we use the dataset coming from the figurines and
% trucks (given)
use_truck_and_car=false;

if (use_truck_and_car)
    load(['datasets' filesep 'norb_binary_truckcar']);
    train_cat_s=double(2*train_cat_s-7); % change 1 and 3 to -1 and 1
    test_cat_s=double(2*test_cat_s-7);
else
    load(['datasets' filesep 'norb_binary']);
    train_cat_s=double(train_cat_s-2); % change 1 and 3 to -1 and 1
    test_cat_s=double(test_cat_s-2);
end
train_left_s=double(train_left_s);
train_right_s=double(train_right_s);


M=576;
H1=10;
H2=5;
K=1;
momentum=0.05;
learning_rate=0.01;

[optimal_weights, error, mu_and_sigmas, missclass, misclassvector, training_error]=trainBinaryMLP(M, H1, H2, train_left_s,train_right_s, train_cat_s,learning_rate,momentum);
[haxes,hline1,hline2] =plotyy(1:length(error),[training_error; error],1:length(error), misclassvector);
axes(haxes(1))
ylabel('Errors on validation and training set')
axes(haxes(2))
ylabel('elements')
legend('correctly classified on val. set','error on training set','error on validation set');

%% test the classifier obtained with the optimal weights on the test set
test_left_norm=normalize(double(test_left_s), mu_and_sigmas(:,1), mu_and_sigmas(:,2));
test_right_norm=normalize(double(test_right_s), mu_and_sigmas(:,3), mu_and_sigmas(:,4));

count_error=0;
countplot=0;
for i=1:size(test_left_norm,2)
    output=mlp(M,H1,H2,test_left_norm(:,i),test_right_norm(:,i),optimal_weights, false);
    class_max=sign(output);
    if (class_max~=test_cat_s(i))
        count_error=count_error+1;
    end
    if ((output>10) && countplot<2)
        figure;
       imshow(reshape(test_left_s(:,i),24,24)');
       title(strcat('output :',num2str(output)));
       countplot=countplot+1;
    end

end

count_error