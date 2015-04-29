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

% Training data: 300 data points from each of the 6 groups (total: 1800)
XTrain1 = data_train(1:300,:);
XTrain2 = data_train(501:800,:);
XTrain3 = data_train(1001:1300,:);
XTrain4 = data_train(1501:1800,:);
XTrain5 = data_train(2001:2300,:);
XTrain6 = data_train(2501:2800,:);

XTrain = double([XTrain1 ; XTrain2 ; XTrain3 ; XTrain4 ; XTrain5 ; XTrain6]);

TTrain1 = labels_train(1:300,:);
TTrain2 = labels_train(501:800,:);
TTrain3 = labels_train(1001:1300,:);
TTrain4 = labels_train(1501:1800,:);
TTrain5 = labels_train(2001:2300,:);
TTrain6 = labels_train(2501:2800,:);

TTrain = double([TTrain1 ; TTrain2 ; TTrain3 ; TTrain4 ; TTrain5 ; TTrain6]);

% Test data: 200 data points from each of the 6 groups (total: 1200)
XTest1 = data_train(301:500,:);
XTest2 = data_train(801:1000,:);
XTest3 = data_train(1301:1500,:);
XTest4 = data_train(1801:2000,:);
XTest5 = data_train(2301:2500,:);
XTest6 = data_train(2801:3000,:);

XTest = double([XTest1 ; XTest2 ; XTest3 ; XTest4 ; XTest5 ; XTest6]);

TTest1 = labels_train(301:500,:);
TTest2 = labels_train(801:1000,:);
TTest3 = labels_train(1301:1500,:);
TTest4 = labels_train(1801:2000,:);
TTest5 = labels_train(2301:2500,:);
TTest6 = labels_train(2801:3000,:);

TTest = double([TTest1 ; TTest2 ; TTest3 ; TTest4 ; TTest5 ; TTest6]);

coef0 = [0, 100, 500, 1000, 2000, 5000, 10000];

% Reach max at 150
numEigenVectors = [20, 50, 100, 120, 150, 160, 180, 200, 400, 600, 800, 2000, 3072];
num = size(numEigenVectors,2);

% PCA on training set
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

