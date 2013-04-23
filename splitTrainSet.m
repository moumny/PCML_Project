function [actL, actR, actC, validL, validR, validC] = splitTrainSet( trainL, trainR, trainC )
%SPLITTRAINSET Split the training set in a validation set and a "actual" training set 
%   Validation set is 1/3 of the training dataset the rest is the actual
%   training set

N = size(trainL,2);
V = floor(1500);

perm = randperm(N);

validL = trainL(:,perm(1:V));
validR = trainR(:,perm(1:V));
validC = trainC(:,perm(1:V));

actL = trainL(:,perm(V+1:end));
actR = trainR(:,perm(V+1:end));
actC = trainC(:,perm(V+1:end));

end

