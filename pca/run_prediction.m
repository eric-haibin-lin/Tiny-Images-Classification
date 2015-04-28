function [y] = run_prediction(final_data_test)
% Give predition y on (arbitrary sized) input test data.
%
%   Inputs:
%       final_data_test: Ni x 3072 test image data.

%   Outputs:
%       y              : Ni x 1 vector of predicted label.
%       (files)        : prediction.csv, mypredictions.mat

load a4data;

% To visualize the first image in the training data
% imshow(reshape(data_train(1,:), 32, 32, 3));

% Convert data to doubles between 0 and 1
% Dn = double(data_nolabel)/255;

%XTrain = data_train;
%TTrain = labels_train;
%XTest = data_test;

XTrain = double(data_train)/255;
TTrain = labels_train;

XTrain2 = double(data_nolabel)/255;
XTest = double(data_test)/255;

XNoLabel = double(data_nolabel)/255;

XPCA = [XTrain ; XNoLabel];
% PCA model on training set, keep all eigenvectors
[base,mean,projX] = pcaimg(XPCA', 3072);

[D, N] = size(XTrain');
[D, Nt] = size(XTest');

X = XTrain' - repmat(mean,1,N);
Xt = XTest' - repmat(mean,1,Nt);

K = 20;
baseK = base(:,1:K);

%zTrain = baseK' * double(X);
%zTest = baseK' * double(Xt);

zTrain = baseK' * X;
zTest = baseK' * Xt;

y = knn_prediction(zTrain, TTrain, 17, zTest);

write_kaggle_csv('prediction', y);

end
