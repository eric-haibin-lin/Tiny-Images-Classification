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

X = double(reshape(XTrain',32,32,3, 1800))/255;
Xt = double(reshape(XTest',32,32,3, 1200))/255;

X1 = reshape(X(:,:,1,:), 32, 32, 1800);
X2 = reshape(X(:,:,2,:), 32, 32, 1800);
X3 = reshape(X(:,:,3,:), 32, 32, 1800);

Xt1 = reshape(Xt(:,:,1,:), 32, 32, 1200);
Xt2 = reshape(Xt(:,:,2,:), 32, 32, 1200);
Xt3 = reshape(Xt(:,:,3,:), 32, 32, 1200);

train_x = 0.21*X1 + 0.72*X2 + 0.07*X3;
test_x = 0.21*Xt1 + 0.72*Xt2 + 0.07*Xt3;

y1 = [ones(1,300) , zeros(1,1500)];
y2 = [zeros(1,300) , ones(1,300), zeros(1,1200)];
y3 = [zeros(1,600) , ones(1,300), zeros(1,900)];
y4 = [zeros(1,900) , ones(1,300), zeros(1,600)];
y5 = [zeros(1,1200) , ones(1,300), zeros(1,300)];
y6 = [zeros(1,1500) , ones(1,300)];

yt1 = [ones(1,200) , zeros(1,1000)];
yt2 = [zeros(1,200) , ones(1,200), zeros(1,800)];
yt3 = [zeros(1,400) , ones(1,200), zeros(1,600)];
yt4 = [zeros(1,600) , ones(1,200), zeros(1,400)];
yt5 = [zeros(1,800) , ones(1,200), zeros(1,200)];
yt6 = [zeros(1,1000) , ones(1,200)];

train_y = double([y1;y2;y3;y4;y5;y6]);
test_y = double([yt1;yt2;yt3;yt4;yt5;yt6]);

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error
rand('state',0)
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
    struct('type', 'c', 'outputmaps', 24, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 1) %subsampling layer
};
cnn = cnnsetup(cnn, train_x, train_y);

opts.alpha = 0.01;
opts.batchsize = 50;
opts.numepochs = 100;

cnn = cnntrain(cnn, train_x, train_y, opts);

%labels = zeros(6, 1200);
[er, bad, labels] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL);

[proxy, final_label] = max(labels);

% labels should be from 0 to 5
final_label = final_label - 1;
accuracy = sum(final_label' == TTest) / 1200;

assert(er<0.12, 'Too big error');