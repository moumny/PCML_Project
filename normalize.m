function [ norm, mu, sigma ] = normalize( input, mu, sigma )
% Normalize the input, through rows
% if mu and sigma are not specified, then function relies on the input to
% determine them
% if they are specified, they are used.


if (nargin~=3)
mu = mean(input,2);
sigma = std(input,1,2);
end

diff = input - repmat(mu,1, size(input, 2));
norm = diff ./ repmat(sigma,1,size(input,2));

end

