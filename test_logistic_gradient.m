N=4;
rand_train=randn(N,1);
rand_t=[0;0;1;0;0];

W_logistic=randn(5,N)/N;
sigma_k=exp(W_logistic*rand_train);
sigma_k=sigma_k/sum(sigma_k);
grad_v=(sigma_k-rand_t)*rand_train';

eps=0.0001;
grad_v_num=zeros(5,N);
for i=1:N*5
    d=zeros(5,N);
    d(i)=1;
    
    lsexp_y=log(sum(exp((W_logistic+eps*d)*rand_train),1));
    error_plus=lsexp_y-sum(rand_t.*((W_logistic+eps*d)*rand_train));
    lsexp_y=log(sum(exp((W_logistic-eps*d)*rand_train),1));
    error_minus=lsexp_y-sum(rand_t.*((W_logistic-eps*d)*rand_train));
    grad_v_num(i)=(error_plus-error_minus)/2/eps;
end

grad_v-grad_v_num
