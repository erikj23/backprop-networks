r = neural_network;
r.initialize(1, 2, @logsig);
r.add_layer(3,2,@logsig);

% build training input matrix
p1=[-1	1	1	1	1	-1	1	-1	-1	-1	-1	1	1	-1	-1	-1	-1	1	1	-1	-1	-1	-1	1	-1	1	1	1	1	-1]';
p2=[ -1	-1	-1	-1	-1	-1	1 -1 -1	-1	-1	-1	1	1	1	1	1	1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1]';
p3=[1 -1 -1	-1	-1	-1	1 -1 -1	1	1	1	1	-1	-1	1	-1	1	-1	1	1	-1	-1	1	-1	-1	-1	-1	-1	1]';
P=[p1 p2 p3];

% build expected output matrix
t1=[1 0 0]';
t2=[0 1 0]';
t3=[0 0 1]';
T=[t1 t2 t3];

training_set = {P,T};

r.train(1, {{[1], 1+sin(pi/4)}});