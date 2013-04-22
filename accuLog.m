function [ y ] = accuLog( x )
%ACCULOG Computes accuratly log(1 + exp(x))
%   Detailed explanation goes here

if x < 0
    y = log1p(exp(x));
else
    y = x + log1p(exp(-x));

end

