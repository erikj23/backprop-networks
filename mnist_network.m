
% reset environment
clear
close all

% load images & labels
images = load_images('..\..\Downloads\train-images-idx3-ubyte');
labels = load_labels('..\..\Downloads\train-labels-idx1-ubyte');

% format {{p t} ...}
samples = length(images);
sample_set = cell(1, samples);
for i=1:samples
    sample_set{i} = {images(:, i) labels(i)};
end

% shuffle sample order
sample_set = sample_set(randperm(length(sample_set)));

% split samples into a training & validation set
testing = 0.70;
validation = 0.30;
testing_set = sample_set(1:samples*testing);
validation_set = sample_set(1:samples*validation);

% hyper-parameters
epochs = 100;
batch_size = length(validation_set);
sample = sample_set{1};
inputs = length(sample{1});
outputs = length(sample{2});
first_layers = [10];

for neurons=first_layers
    % pre-computation graph configuration
    figure;
    subplot(2, 1, 1);
    hold on; grid on;
    title('mean squared error:epochs')
    xlabel('epochs')
    ylabel('mse')
    
    % create & train a network
    r = neural_network;
    
    % first layer has inputs equivalent to input pattern
    r.initialize(inputs, neurons, @logsig);
    
    % last layer has neurons equivalent to output target
    r.add_layer(outputs, @purelin)
    
    % time the performance
    tic
    mse = r.train(epochs, batch_size, validation_set);
    toc
        
    % post-computation graph configuration
    plot(1:epochs, mse)
    subplot(2, 1, 2)
    imagesc(r.layers{end}.w);
    colormap(hsv);
    colorbar;
    title('final weight matrix');
end