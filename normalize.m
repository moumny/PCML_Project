function [ norm, mu, sigma ] = normalize( input )
%NORMALIZE Summary of this function goes here
%   Detailed explanation goes here

mu = mean(input,2);

diff = input - repmat(mu,1, size(input, 2));
sigma = std(input,1,2);
norm = diff ./ repmat(sigma,1,size(input,2));


end

