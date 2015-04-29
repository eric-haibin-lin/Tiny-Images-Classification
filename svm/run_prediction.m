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



%train with all data available, for submission
%model = svmtrain(double(labels_train), double(data_train), '-t 1');
%[predict_label] = svmpredict(double(zeros(1200, 1)), double(data_test), model); 



%to tune SVM parameters
coef0 = [0.001, 0.01, 0.1, 0, 1];
gammas = [0.0001, 0.001, 0.01, 0.1, 1];
paramToTune = gammas;
sizes = size(paramToTune, 2);

accuracyTrainArr = zeros(1,sizes);
accuracyTestArr = zeros(1,sizes);

for index = 1:sizes
    
    model = svmtrain(TTrain, XTrain, horzcat(['-t 1 -d 3 -g', ' ', int2str(paramToTune(index))]));
    [predict_labelTrain, accuracyTrain, dec_valuesTrain] = svmpredict(TTrain, XTrain, model); 
    [predict_labelTest, accuracyTest, dec_valuesTest] = svmpredict(TTest, XTest, model); 
 
    accuracyTrainArr(index) = accuracyTrain(1);
    accuracyTestArr(index) = accuracyTest(1);
    
end 


plot(paramToTune, accuracyTrainArr, paramToTune, accuracyTestArr); 
title('poly kernal degree 3 default coef, gamma 0.0001 ~ 1, all default settings, no PCA');
legend('train', 'test');
ylim([40 47]);



%write_kaggle_csv('prediction', y);

%end
