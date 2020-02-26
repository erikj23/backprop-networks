clear
close all

%% Build training input matrix
p1=[-1    1    1    1    1    -1    1    -1    -1    -1    -1    1    1    -1    -1    -1    -1    1    1    -1    -1    -1    -1    1    -1    1    1    1    1    -1]';
p2=[ -1    -1    -1    -1    -1    -1    1 -1 -1    -1    -1    -1    1    1    1    1    1    1    -1    -1    -1    -1    -1    -1    -1    -1    -1    -1    -1    -1]';
p3=[1 -1 -1    -1    -1    -1    1 -1 -1    1    1    1    1    -1    -1    1    -1    1    -1    1    1    -1    -1    1    -1    -1    -1    -1    -1    1]';
P=[p1 p2 p3];

%% Build expected output matrix
t1=[1 0 0]';
t2=[0 1 0]';
t3=[0 0 1]';
T=[t1 t2 t3];

%% Build training and test sets
training_set = {{p1 t1} {p2 t2} {p3 t3}};
test_set = {{p1 t1} {p2 t2} {p3 t3}};

%% Instantiate network
r = neural_network;

% length = 30 neurons = 2
n_neurons=length(p1);
r.initialize(n_neurons, 3, @logsig);

% t = 3 so last layer has 3 neurons
r.add_layer(3, @logsig);

%% Train network
tic
r.train(50, 1, training_set);
toc

%% Test network on test_set
correct = r.test(test_set);
results = r.verify(test_set);

%% Display results of network on original ("clean") inputs
% for i=1:length(results)
%     disp([results{i}{1},results{i}{2}]);
% end

%% Plot final weight matrix
% imagesc(r.layers{end}.w);
% colormap(hsv);
% colorbar;
% title('Final Weight Matrix for output layer of Warmup-Network');
% 
% imagesc(r.layers{end-1}.w);
% colormap(hsv);
% colorbar;
% title('Final Weight Matrix for hidden layer of Warmup-Network');

%%  Plot accuracy rates for noisy digits
experiments=3;
accuracy=r.test_noisy(test_set,experiments);

clf
plot([0, 4, 8], accuracy,'LineWidth',1.5)
hold on

%Set Axis Ticks, Axis Labels, and Legend
axis([0 8 0 100])
x = linspace(0,8,3);
set(gca,'xtick',x);
xticklabels({'0','4','8'})
% xticklabels({'2','3','4','5','6','7'})
% legend('2 occluded pixels', '4 occluded pixels', '6 occluded pixels', 'location','best')
title('Accuracy Rate for Occluded Pixels in Digits')
xlabel('# of Occluded Pixels')
ylabel('% Error in test')