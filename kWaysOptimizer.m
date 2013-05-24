%% this file try to find the best parameters for the 5 ways MLP
%% general settings 
load(['datasets' filesep 'norb_5class']);
train_cat_s=double(train_cat_s); 
train_left_s=double(train_left_s);
train_right_s=double(train_right_s);
M=576;
K=5;

%% Optimization in terms of H1 and H2
% for this part, momentum and learning rate are fixed
learning_rate=0.01;
momentum=0.1;
%Note à Younes : j'avais observé que la convergence était rapide avec ce
%learning rate, et que le momentum lissait un peu la courbe d'erreur

% the early stopping criteria is still the sliding window (but that can be
% changed directly in trainMultiMLP.m)

%possibles values for H1 and H2
H1s=[2,5,8,10,12,15,20,25,50,75];
H2s=[2,3,5,8,10,12,20,30];
% number of redondancy (to avoid random effect)
red=10;
% the matrix imn which the results will be stored
average_error=zeros(length(H1s),length(H2s));
std_on_error=zeros(length(H1s),length(H2s));
nb_epochs=zeros(length(H1s),length(H2s));
 % just for info : nb of misclassification on validation set
nb_missclassification=zeros(length(H1s),length(H2s));

for i=1:length(H1s)
   H1=H1s(i);
   for j=1:length(H2s)
       H2=H2s(j);
       if H2>H1
           % we limit the optimization for 
           % H2>H1
           break;
       end
       disp(strcat('optimization for H1=',num2str(H1),' and H2=',num2str(H2)));
       vect_error=zeros(red,1);
       vect_misclass=zeros(red,1);
       vect_epochs=zeros(red,1);
       for k=1:red
        [optimal_weights, error, mu_and_sigmas, misclass]=trainMultiMLP(M, H1, H2, K, train_left_s,train_right_s, train_cat_s, learning_rate,momentum);
        vect_error(k)=error(length(error));
        vect_epochs(k)=length(error);
        vect_misclass(k)=misclass;
       end
       average_error(i,j)=mean(vect_error);
       std_on_error(i,j)=std(vect_error);
       nb_epochs(i,j)=mean(vect_epochs);
       nb_missclassification(i,j)=mean(vect_misclass);
   end
   save('average_error','average_error');
   save('std_on_error','std_on_error');
   save('nb_epochs','nb_epochs');
   save('nb_missclassification','nb_missclassification');
end

%% learning rate, momentum

lrs=[0.001,0.005,0.01,0.05];
moms=[0,0.01,0.05,0.1];
H1=60;
H2=40;
red=1;
% the matrix imn which the results will be stored
average_error_lr=zeros(length(lrs),length(moms));

for i=1:length(lrs)
   lr=lrs(i);
   for j=1:length(moms)
       mom=moms(j);
       disp(strcat('optimization for lr=',num2str(lr),' and mom=',num2str(mom)));
       vect_error=zeros(red,1);
       for k=1:red
        [optimal_weights, error, mu_and_sigmas, misclass]=trainMultiMLP(M, H1, H2, K, train_left_s,train_right_s, train_cat_s, lr,mom);
        vect_error(k)=error(length(error));
       end
       average_error_lr(i,j)=mean(vect_error);
   end
   save('average_error_lr','average_error_lr');
end
