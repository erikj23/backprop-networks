
%% reset environment
clear
close all

%% load images, labels, & perform preprocessing
[sample_set, samples] = create_sample_set('../../downloads/FMNIST/train.csv');

%% split samples into a training & validation set
training = samples*0.70;
validation = samples*0.30;
training_set = sample_set(1:training);
validation_set = sample_set(training+1:training+validation);

%% set hyper-parameters
epochs = 2;
batch_size = 100;
sample = sample_set{1};
input_size = length(sample{1});
input_neurons = 10;
output_neurons = length(sample{2});

%% train network
r = neural_network;

% first layer has inputs equivalent to input pattern
r.initialize(input_size, input_neurons, @logsig);

% last layer has neurons equivalent to output target
r.add_layer(output_neurons, @softmax)

% train & time the performance
tic
[accuracy_t, accuracy_v, loss] = r.train(epochs, batch_size, training_set, validation_set);
toc

%% graph results
figure;


% accuracy
subplot(3, 1, 1);
hold on;
plot(1:epochs, accuracy_t)
plot(1:epochs, accuracy_v)
title('accuracy:epochs')
ylabel('% accuracy')
xlabel('epochs')

% mse
subplot(3, 1, 2);
plot(1:epochs, loss)
title('mean squared error:epochs')
ylabel('error')
xlabel('epochs')

% weight matrix
subplot(3, 1, 3);
imagesc(r.layers{end}.w);
colormap(hsv);
colorbar;
title('final weight matrix');

%% follow up with validation
%accuracy_t = r.test(training_set) / training * 100;
%accuracy_v = r.test(validation_set) / validation * 100;

%% finish with testing if validation was decent
%[ids, images, samples] = load_fmnist_tests('../../downloads/FMNIST/test.csv');
%predictions = r.kaggle(images);
%output=table(ids', predictions');
%output.Properties.VariableNames = {'Id' 'label'};
%writetable(output, sprintf('%s-L%d-E%d-B%d-A%0.2f.csv', 'base', length(r.layers), epochs, batch_size, r.alpha))
