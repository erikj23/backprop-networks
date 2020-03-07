
%% reset environment
clear
close all

%% load images, labels, & perform preprocessing
sample_set = create_samples('~/B/downloads/FMNIST/train.csv');

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
input_neurons_list = [100];
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
    
    % create a network
    r = neural_network;
    
    % first layer has inputs equivalent to input pattern
    r.initialize(input_size, input_neurons, @logsig);
    %r.add_layer(10, @logsig)
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

%% follow up with validation
r.test(validation_set)

% 