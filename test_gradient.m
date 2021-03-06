%% this script test that the gradient has been well computed
% using the method detailed p.46 of the book

% We choose an MLP size
M=4; % input's size (Dimension?)
H1=18;
H2=9;
K=5;
XL=randn(M,1);%random input
XR=randn(M,1);

% we choose random weights to start
weights=initializeWeights(M,H1,H2,K);

% for each coordinate, the gradient must be ~ equal
% to [E(w+e)-E(w-e)]/(2e)
epsilon=0.000001;
if  K==1
    ti=1;
    [output, gradient]=mlp(M,H1,H2,XL,XR,weights,true,ti);
else
    ti=zeros(K,1);
    ti(4)=1;
    [output, gradient]=kmlp(M,H1,H2,K,XL,XR,weights,true,ti);
    
end
approximated_gradient=zeros(length(weights),1);
for i=1:length(weights)
    d=zeros(length(weights),1);
    d(i)=epsilon;
    % obtain output of MLP
    if K==1
        aWplusD=mlp(M,H1,H2,XL,XR,weights+d,false,0);
        aWminusD=mlp(M,H1,H2,XL,XR,weights-d,false,0);
        % calculate error function
        EWplusD=log(1+exp(-ti*aWplusD));
        EWminusD=log(1+exp(-ti*aWminusD));
    else
        aWplusD=kmlp(M,H1,H2,K,XL,XR,weights+d,false,0);
        aWminusD=kmlp(M,H1,H2,K,XL,XR,weights-d,false,0);
        % calculmlpate error function
        EWplusD=norm(aWplusD-ti).^2;
        EWminusD=norm(aWminusD-ti).^2;
    end
    approximated_gradient(i)=(EWplusD-EWminusD)/(2*epsilon);
end

%if everything's fine, this must be a vector composed of 0
difference=approximated_gradient-gradient;
error_max=max(difference);
disp('maximum error in gradient');
disp(error_max);

% if needed, show the problems coefficient by coefficient in a matrix way
% to isolate the problem
if (error_max>10*epsilon)
    [W1L, B1L, W1R, B1R, W2L, B2L, W2LR, B2LR, W2R, B2R, W3, B3] = weightsToMatrix(M, H1, H2, K, approximated_gradient-gradient)
end