%function [y] = run_prediction(final_data_test)
% Give predition y on (arbitrary sized) input test data.
%
%   Inputs:
%       final_data_test: Ni x 3072 test image data.

%   Outputs:
%       y              : Ni x 1 vector of predicted label.
%       (files)        : prediction.csv, mypredictions.mat

%load a4data;
% To visualize the first image in the training data
% imshow(reshape(data_train(1,:), 32, 32, 3));
% Convert data to doubles between 0 and 1
%Dn = double(data_nolabel)/255;
%Dt = double(data_train)/255;
%tt = double(labels_train);
%Dv = double(data_test)/255;



clear all;
load a4data;

%train with all data available
XTrain = double(data_train);
TTrain = double(labels_train);
XTest = double(data_test);

K = 150;
[base,mean,projX] = pcaimg(XTrain', K);

[D, Nt] = size(XTest');

Xt = XTest' - repmat(mean,1,Nt);
zTest = base * Xt;

model = svmtrain(TTrain, projX', '-t 1');
[predict_label] = svmpredict(double(zeros(1200, 1)), zTest', model);

% write_kaggle_csv('prediction', predict_label);

%{
% Reach max at 150
numEigenVectors = [20, 50, 100, 120, 150, 160, 180, 200, 400, 600, 800];
num = size(numEigenVectors,2);

[base,mean,projX] = pcaimg(XTrain', 3072);

[D, N] = size(XTrain');
[D, Nt] = size(XTest');

X = XTrain' - repmat(mean,1,N);
Xt = XTest' - repmat(mean,1,Nt);

accuracyTrainArr = zeros(1,num);
accuracyTestArr = zeros(1,num);

for index = 1:num
    K = numEigenVectors(index);
    baseK = base(:,1:K);

    zTrain = baseK' * X;
    zTest = baseK' * Xt;
    
    %model = svmtrain(TTrain, XTrain, horzcat(['-t 1 -d 3 -r', ' ', int2str(coef0(index))]));
    model = svmtrain(TTrain, zTrain', '-t 1');
    [predict_labelTrain, accuracyTrain, dec_valuesTrain] = svmpredict(TTrain, zTrain', model); 
    [predict_labelTest, accuracyTest, dec_valuesTest] = svmpredict(TTest, zTest', model); 
  
    accuracyTrainArr(index) = accuracyTrain(1);
    accuracyTestArr(index) = accuracyTest(1);
end 

% Plot accuracy
figure(2);
hold on;
plot(numEigenVectors, accuracyTrainArr, 'r', 'LineWidth', 3); 
plot(numEigenVectors, accuracyTestArr, 'k', 'LineWidth', 3); 

xlabel('Number of Eigenvectors');
ylabel('Classification accuracy');
legend('Training set', 'Test set');
%}

%{
plot(coef0, accuracyTrainArr, coef0, accuracyValidArr, coef0, accuracyTestArr, coef0, accuracyComboArr); 
title('poly kernal degree 3 default gamma, coef 0 ~ 10000, all default settings, no PCA');
legend('train', 'valid', 'test', 'combo');
%ylim([40 47]);
%}


%write_kaggle_csv('prediction', y);
