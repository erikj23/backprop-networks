
%% reset environment
clear
close all

%% load images, labels, & perform preprocessing
[sample_set, samples] = create_sample_set('C:\Users\Mike\Documents\css-485\hw\backprop-networks\train.csv');
[test_set, test_examples] = create_sample_set('C:\Users\Mike\Documents\css-485\hw\backprop-networks\test.csv');

%% split samples into a training & validation set
training = samples*0.70;
validation = samples*0.30;
training_set = sample_set(1:training);
validation_set = sample_set(training:training+validation);

%% set hyper-parameters
epochs = 10;
batch_size = 100;
sample = sample_set{1};
input_size = length(sample{1});
input_neurons_list = [10];
output_neurons = length(sample{2});

%% train & graph results
for input_neurons=input_neurons_list
    % pre-computation graph configuration
    figure;
    subplot(2, 1, 1);
    hold on; grid on;
    title('mean squared error:epochs')
    xlabel('epochs')
    ylabel('accuracy %')
    
    % create a network
    r = neural_network;
    
    % first layer has inputs equivalent to input pattern
    r.initialize(input_size, input_neurons, @logsig);
    %r.add_layer(10, @logsig)
    % last layer has neurons equivalent to output target
    r.add_layer(output_neurons, @sft_max)
    
    % train & time the performance
    tic
    accuracy_rates = r.train(epochs, batch_size, training_set);
    toc
    
    % post-computation graph configuration
    plot(1:epochs, accuracy_rates)
    subplot(2, 1, 2)
    imagesc(r.layers{end}.w);
    colormap(hsv);
    colorbar;
    title('final weight matrix');
end

%% follow up with validation
training
size(training_set)
size(validation_set)
size(test_set)
tr_accuracy = r.test(training_set) / training * 100
v_accuracy = r.test(validation_set) / validation * 100



%% finish with testing if validation was decent
[ids, images, samples] = load_fmnist_tests('C:\Users\Mike\Documents\css-485\hw\backprop-networks\test.csv');
predictions = r.kaggle(images);
output=table(ids', predictions');
output.Properties.VariableNames = {'Id' 'label'};
writetable(output, sprintf('%s-L%d-E%d-B%d-A%0.2f.csv', 'base', length(r.layers), epochs, batch_size, r.alpha))