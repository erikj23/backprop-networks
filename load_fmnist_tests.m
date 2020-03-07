function [images, ids] = load_fmnist_tests(filename)
    % read matrix into memory and transpose
    data = readmatrix(filename)';
    
    % row 2 contains the labels
    ids = data(1, :);
    
    % delete row 1
    data(1, :) = [];
    
    % remaining columns are 28x28 images in column vector form
    images = data;
end