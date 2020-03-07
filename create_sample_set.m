function [sample_set, samples] = create_sample_set(filename)
    %% load images & labels
    [images, digit_labels] = load_fmnist_samples(filename);

    %% get number of samples
    samples = length(images);

    %% replace digits with vectors
    vector_labels = cell(1, samples);
    for i=1:samples
        vector_labels{i} = dec2vec(digit_labels(i));
    end

    %% convert data to the following format {{p t} ...}
    sample_set = cell(1, samples);
    for i=1:samples
        sample_set{i} = {images(:, i) vector_labels{i}};
    end

    %% shuffle sample order
    sample_set = sample_set(randperm(length(sample_set)));
end