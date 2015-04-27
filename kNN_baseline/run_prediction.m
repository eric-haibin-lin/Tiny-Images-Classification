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
Dn = double(data_nolabel)/255;

Dt = double(data_train)/255;
tt = double(labels_train);

Dv = double(data_test)/255;

y = knn_prediction(Dt', tt, 17, Dv');

write_kaggle_csv('prediction', y);

end
