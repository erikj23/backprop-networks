
%% reset enviornment
clear

%% create training_set
p = 1;
t = 1+sin(pi/4);
training_set = {{p t}};

%% set hyper-parameters
epochs = 1;
batch_size = 1;
input_size = 1;
input_neurons = 2;
output_neurons = 1;

%% create network
r = neural_network;
r.initialize(input_size, input_neurons, @logsig);
r.add_layer(output_neurons, @purelin);

%% pre-set initial weights & biases for layer 1
r.layers{1}.w = [-0.27 -0.41]';
r.layers{1}.b = [-0.48 -0.13]';

%% pre-set initial weights & biases for layer 1
r.layers{2}.w = [0.09 -0.17];
r.layers{2}.b = 0.48;

%% train network
tic
r.train(epochs, batch_size, training_set);
toc