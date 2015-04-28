clear all;

load a4data;

% Training data: 200 data points from each of the 6 groups (total: 1200)
XTrain1 = data_train(1:200,:);
XTrain2 = data_train(501:700,:);
XTrain3 = data_train(1001:1200,:);
XTrain4 = data_train(1501:1700,:);
XTrain5 = data_train(2001:2200,:);
XTrain6 = data_train(2501:2700,:);

XTrain = im2double([XTrain1 ; XTrain2 ; XTrain3 ; XTrain4 ; XTrain5 ; XTrain6]);

TTrain1 = labels_train(1:200,:);
TTrain2 = labels_train(501:700,:);
TTrain3 = labels_train(1001:1200,:);
TTrain4 = labels_train(1501:1700,:);
TTrain5 = labels_train(2001:2200,:);
TTrain6 = labels_train(2501:2700,:);

TTrain = [TTrain1 ; TTrain2 ; TTrain3 ; TTrain4 ; TTrain5 ; TTrain6];

% Validation data: 150 data points from each of the 6 groups (total: 900)
XValid1 = data_train(201:350,:);
XValid2 = data_train(701:850,:);
XValid3 = data_train(1201:1350,:);
XValid4 = data_train(1701:1850,:);
XValid5 = data_train(2201:2350,:);
XValid6 = data_train(2701:2850,:);

XValid = im2double([XValid1 ; XValid2 ; XValid3 ; XValid4 ; XValid5 ; XValid6]);

TValid1 = labels_train(201:350,:);
TValid2 = labels_train(701:850,:);
TValid3 = labels_train(1201:1350,:);
TValid4 = labels_train(1701:1850,:);
TValid5 = labels_train(2201:2350,:);
TValid6 = labels_train(2701:2850,:);

TValid = [TValid1 ; TValid2 ; TValid3 ; TValid4 ; TValid5 ; TValid6];

% Test data: 150 data points from each of the 6 groups (total: 900)
XTest1 = data_train(351:500,:);
XTest2 = data_train(851:1000,:);
XTest3 = data_train(1351:1500,:);
XTest4 = data_train(1851:2000,:);
XTest5 = data_train(2351:2500,:);
XTest6 = data_train(2851:3000,:);

XTest = im2double([XTest1 ; XTest2 ; XTest3 ; XTest4 ; XTest5 ; XTest6]);

TTest1 = labels_train(351:500,:);
TTest2 = labels_train(851:1000,:);
TTest3 = labels_train(1351:1500,:);
TTest4 = labels_train(1851:2000,:);
TTest5 = labels_train(2351:2500,:);
TTest6 = labels_train(2851:3000,:);

TTest = [TTest1 ; TTest2 ; TTest3 ; TTest4 ; TTest5 ; TTest6];

% PCA model on training set, keep all eigenvectors
[base,mean,projX] = pcaimg(XTrain', 3072);

num = 10;
errorValidation = zeros(1, num);
errorTest = zeros(1, num);
numEigenVectors = [20, 100, 200, 400, 600, 800, 1200, 1600, 2000, 3072];

[D, N] = size(XTrain');
[D, Nv] = size(XValid');
[D, Nt] = size(XTest');

X = XTrain' - repmat(mean,1,N);
Xv = XValid' - repmat(mean,1,Nv);
Xt = XTest' - repmat(mean,1,Nt);

k = 17;

for i = 1:num
  K = numEigenVectors(i);

  baseK = base(:,1:K);

  %zTrain = baseK' * double(X);
  %zValid = baseK' * double(Xv);
  %zTest = baseK' * double(Xt);

  zTrain = baseK' * X;
  zValid = baseK' * Xv;
  zTest = baseK' * Xt;

  yV = knn_prediction(zTrain, TTrain, k, zValid);
  yT = knn_prediction(zTrain, TTrain, k, zTest);
  
  accValidation(i) = sum((yV > 0.5) - TValid == 0)/Nv;
  accTest(i) = sum((yT > 0.5) - TTest == 0)/Nt;
end 

% Plot accuracy
figure(2);
hold on;
plot(numEigenVectors, accValidation, 'r', 'LineWidth', 3); 
plot(numEigenVectors, accTest, 'k', 'LineWidth', 3); 

xlabel('Number of Eigenvectors');
ylabel('Classification accuracy');
legend('Validation set', 'Test set');


