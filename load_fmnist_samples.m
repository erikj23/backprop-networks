function [images, labels] = load_fmnist_samples(filename)
    % read matrix into memory and transpose
    data = readmatrix(filename)';
   
    % row 2 contains the labels
    labels = data(2, :);
    
    % delete row 1 and row 2, row 1 contains ID
    data([1, 2], :) = [];
    
    % remaining columns are 28x28 images in column vector form
    images = data;
end