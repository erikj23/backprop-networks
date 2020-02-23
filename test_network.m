clear
close all

r = neural_network;
r.initialize(1, 2, @logsig);
r.add_layer(1, @purelin);

r.layers{1}.w = [-0.27 -0.41]';
r.layers{1}.b = [-0.48 -0.13]';
r.layers{1}.n = 0;
r.layers{1}.a = 0;

r.layers{2}.w = [0.09 -0.17];
r.layers{2}.b = [0.48];
r.layers{2}.n = 0;
r.layers{2}.a = 0;

r.train(2, 2, {{[1] 1+sin(pi/4)} {[1] 1+sin(pi/4)}});