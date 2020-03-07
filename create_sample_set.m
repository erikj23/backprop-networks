function [sample_set, samples] = create_sample_set(filename)
    %% load images & labels
    [images, digit_labels] = load_fmnist_samples(filename);

    %% get number of samples
    samples = length(images);

    %% replace digits with vectors
    labels_cell = cell(1, samples);
    for i=1:samples
        labels_cell{i} = dec2vec(digit_labels(i));
    end

    %% bind p-t to the following format {{p t} ...}
    sample_set = cell(1, samples);
    for i=1:samples
        sample_set{i} = {images(:, i) labels_cell{i}};
    end

    %% shuffle sample order
    sample_set = sample_set(randperm(length(sample_set)));
    
    %% restore data in into matrices for gpu operations
    labels_matrix=zeros(size(sample_set{1}{2}, 1), samples);
    for i=1:samples
        p = sample_set{i}{1};
        t = sample_set{i}{2};
        images(:, i) = p;
        labels_matrix(:, i) = t;
    end
end