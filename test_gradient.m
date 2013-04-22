%% this script test that the gradient has been well computed
% using the method detailed p.46 of the book

% We choose an MLP size
M=4; % input's size (Dimension?)
H1=3;
H2=2;
XL=randn(M,1);%random input
XR=randn(M,1);

% we choose random weights to start
weights=initializeWeights(M,H1,H2);

% for each coordinate, the gradient must be ~ equal
% to [E(w+e)-E(w-e)]/(2e)
epsilon=0.000001;
ti=1;
[output, gradient]=mlp(M,H1,H2,XL,XR,weights,true,ti);

approximated_gradient=zeros(length(weights),1);
for i=1:length(weights)
    d=zeros(length(weights),1);
    d(i)=epsilon;
    % obtain output of MLP
    aWplusD=mlp(M,H1,H2,XL,XR,weights+d,false,0);
    aWminusD=mlp(M,H1,H2,XL,XR,weights-d,false,0);
    % calculate error function
    EWplusD=log(1+exp(-ti*aWplusD));
    EWminusD=log(1+exp(-ti*aWminusD));
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
    [W1L, B1L, W1R, B1R, W2L, B2L, W2LR, B2LR, W2R, B2R, W3, B3] = weightsToMatrix(M, H1, H2, approximated_gradient-gradient)
end