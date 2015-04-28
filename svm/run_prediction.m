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

% Training data: 200 data points from each of the 6 groups (total: 1200)
XTrain1 = data_train(1:200,:);
XTrain2 = data_train(501:700,:);
XTrain3 = data_train(1001:1200,:);
XTrain4 = data_train(1501:1700,:);
XTrain5 = data_train(2001:2200,:);
XTrain6 = data_train(2501:2700,:);

XTrain = double([XTrain1 ; XTrain2 ; XTrain3 ; XTrain4 ; XTrain5 ; XTrain6]);

TTrain1 = labels_train(1:200,:);
TTrain2 = labels_train(501:700,:);
TTrain3 = labels_train(1001:1200,:);
TTrain4 = labels_train(1501:1700,:);
TTrain5 = labels_train(2001:2200,:);
TTrain6 = labels_train(2501:2700,:);

TTrain = double([TTrain1 ; TTrain2 ; TTrain3 ; TTrain4 ; TTrain5 ; TTrain6]);

% Validation data: 150 data points from each of the 6 groups (total: 900)
XValid1 = data_train(201:350,:);
XValid2 = data_train(701:850,:);
XValid3 = data_train(1201:1350,:);
XValid4 = data_train(1701:1850,:);
XValid5 = data_train(2201:2350,:);
XValid6 = data_train(2701:2850,:);

XValid = double([XValid1 ; XValid2 ; XValid3 ; XValid4 ; XValid5 ; XValid6]);

TValid1 = labels_train(201:350,:);
TValid2 = labels_train(701:850,:);
TValid3 = labels_train(1201:1350,:);
TValid4 = labels_train(1701:1850,:);
TValid5 = labels_train(2201:2350,:);
TValid6 = labels_train(2701:2850,:);

TValid = double([TValid1 ; TValid2 ; TValid3 ; TValid4 ; TValid5 ; TValid6]);

% Test data: 150 data points from each of the 6 groups (total: 900)
XTest1 = data_train(351:500,:);
XTest2 = data_train(851:1000,:);
XTest3 = data_train(1351:1500,:);
XTest4 = data_train(1851:2000,:);
XTest5 = data_train(2351:2500,:);
XTest6 = data_train(2851:3000,:);

XTest = double([XTest1 ; XTest2 ; XTest3 ; XTest4 ; XTest5 ; XTest6]);

TTest1 = labels_train(351:500,:);
TTest2 = labels_train(851:1000,:);
TTest3 = labels_train(1351:1500,:);
TTest4 = labels_train(1851:2000,:);
TTest5 = labels_train(2351:2500,:);
TTest6 = labels_train(2851:3000,:);

TTest = double([TTest1 ; TTest2 ; TTest3 ; TTest4 ; TTest5 ; TTest6]);

%combines validation set and test set.
XCombo = [XValid; XTest];
TCombo = [TValid; TTest];


coef0 = [0, 100, 500, 1000, 2000, 5000, 10000];
%gammas = [0.0001, 0.001, 0.01, 0.1, 1];

sizes = size(coef0, 2);

%train with all data available
model = svmtrain(double(labels_train), double(data_train), '-t 1');
[predict_label] = svmpredict(double(zeros(1200, 1)), double(data_test), model); 

write_kaggle_csv('prediction', predict_label);

%{
accuracyTrainArr = zeros(1,sizes);
accuracyValidArr = zeros(1,sizes);
accuracyTestArr = zeros(1,sizes);
accuracyComboArr = zeros(1,sizes);

for index = 1:sizes
    
    model = svmtrain(TTrain, XTrain, horzcat(['-t 1 -d 3 -r', ' ', int2str(coef0(index))]));
    [predict_labelTrain, accuracyTrain, dec_valuesTrain] = svmpredict(TTrain, XTrain, model); 
    [predict_labelValid, accuracyValid, dec_valuesValid] = svmpredict(TValid, XValid, model); 
    [predict_labelTest, accuracyTest, dec_valuesTest] = svmpredict(TTest, XTest, model); 
    [predict_labelCombo, accuracyCombo, dec_valuesCombo] = svmpredict(TCombo, XCombo, model); 

    accuracyTrainArr(index) = accuracyTrain(1);
    accuracyValidArr(index) = accuracyValid(1);
    accuracyTestArr(index) = accuracyTest(1);
    accuracyComboArr(index) = accuracyCombo(1);
    
end 


plot(coef0, accuracyTrainArr, coef0, accuracyValidArr, coef0, accuracyTestArr, coef0, accuracyComboArr); 
title('poly kernal degree 3 default gamma, coef 0 ~ 10000, all default settings, no PCA');
legend('train', 'valid', 'test', 'combo');
%ylim([40 47]);
%}



%write_kaggle_csv('prediction', y);

%end
