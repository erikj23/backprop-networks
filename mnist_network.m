
%% reset environment
clear
close all

%% load images & labels
images = load_images('..\..\Downloads\train-images-idx3-ubyte');
number_labels = load_labels('..\..\Downloads\train-labels-idx1-ubyte');

%% get number of samples
samples = length(images);

%% replace digits with vectors
vector_labels = cell(1, samples);
for i=1:samples
    vector_labels{i} = dec2vec(number_labels(i));
end

%% convert data to the following format {{p t} ...}
sample_set = cell(1, samples);
for i=1:samples
    sample_set{i} = {images(:, i) vector_labels{i}};
end

%% shuffle sample order
sample_set = sample_set(randperm(length(sample_set)));

%% split samples into a training & validation set
training = 0.70;
validation = 0.30;
training_set = sample_set(1:samples*training);
validation_set = sample_set(1:samples*validation);

%% set hyper-parameters
epochs = 100;
batch_size = 100;
sample = sample_set{1};
input_size = length(sample{1});
input_neurons_list = [1 3 5 7 10 100];
output_neurons = length(sample{2});

%% train & graph results
for input_neurons=input_neurons_list
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
    r.initialize(input_size, input_neurons, @logsig);
    
    % last layer has neurons equivalent to output target
    r.add_layer(output_neurons, @logsig)
    
    % train & time the performance
    tic
    mse = r.train(epochs, batch_size, training_set);
    toc
        
    % post-computation graph configuration
    plot(1:epochs, mse)
    subplot(2, 1, 2)
    imagesc(r.layers{end}.w);
    colormap(hsv);
    colorbar;
    title('final weight matrix');
end
