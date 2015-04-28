function write_kaggle_csv( csv_name, labels )
% Write labels to [csv_name].csv for kaggle submission.
%
%   Inputs:
%       csv_name: output csv filename (without ".csv").
%       labels:   N*1 vector of the result labels.
%   Outputs:
%       file [csv_name].csv

f = fopen(strcat(csv_name,'.csv'), 'w');
fprintf(f, 'id,label\n');
index = [1:length(labels)]';
dlmwrite(f, [index,uint32(labels)], ',', '-append' );
fclose(f);
end
