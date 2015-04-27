function [ y ] = knn_prediction( Xtrain, ttrain, K, X )
% Calculate the output of the KNN classifier.
%   
%   Inputs:
%       Xtrain:  M x N training data matrix, each column is a data point.
%       ttrain:  N x 1 training target vector.
%       K:  number of neighbours to use.
%       X:  M x Ni input data matrix (assume Ni number of inputs).                     
%
%   Outputs:
%       y: Ni x 1 vector of predicted label.

D = l2_distance(X, Xtrain); 
[tmp, Index] = sort(D,2);

KNN = Index(:,1:K); 
y = mode(ttrain(KNN),2); 
end

