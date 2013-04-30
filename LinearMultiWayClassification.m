% In this file, we implement linear multiway classification
%% load file and create dataset
load(['datasets' filesep 'norb_5class']);
train_x=[double(train_left_s);double(train_right_s)];
train_t=-1*ones(5,size(train_cat_s,2));
for i=1:size(train_cat_s,2)
   train_t(double(train_cat_s(i))+1,i)=1; 
end
test_x=[double(test_left_s); double(test_right_s)];
test_cat=double(test_cat_s)+1;

%% Using a squarred error
% There exists a global solution. No need for cross - validation or so,
% the evaluation will be done on the test set. 
% In fact we find 5 vector w, one for every coordinate of the output
% I took notations of course
W_squared_error=zeros(5,size(train_x,1));
for i=1:5
    t=train_t(i,:)';
    Phi=train_x';
    w=(Phi'*Phi)\Phi'*t;
    W_squared_error(i,:)=w';
end

%% Then we test the obtained solutions on the test set
count_squared_error=0;
for i=1:size(test_x,2)
    t_squared_error=W_squared_error*test_x(:,i);
    [~, class_squared_error]=max(t_squared_error);
    if (class_squared_error~=test_cat(i))
        count_squared_error=count_squared_error+1;
    end
end
disp('error with linear classifiers');
disp(strcat('with squared eror function ',num2str(count_squared_error), ... 
    ' misclassified items on ',num2str(size(test_cat,2))));