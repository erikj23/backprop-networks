%% NEURAL_NETWORK custom neural network implementation
% 
% PROPERTIES
%   alpha - learning rate
%   layers - neural network layers
classdef neural_network < handle

    properties(Access='private')
        initialized = false;
    end 
    properties%(SetAccess='private')
        alpha = 0.1;
        layers = {};
    end 
    
    methods        
        
        %
        % INITIALIZE sets first layer of network
        %
        % SYNOPSIS initialize(self, inputs, neurons, transfer_function)
        % INPUT inputs: length of initial input vector
        % INPUT neurons: desired neurons in layer
        % INPUT transfer_function: transfer_function for layer
        %
        function initialize(self, inputs, neurons, transfer_function)
            new_layer = layer;
            new_layer.initialize(neurons, inputs, transfer_function);
            self.layers{end + 1} = new_layer;
            self.initialized = true;
        end
        
        %
        % ADD_LAYER adds a layer to the network
        %
        % SYNOPSIS add_layer(self, neurons, transfer_function)
        %   the network must be initialized
        %
        % INPUT neurons: desired neurons in layer
        % INPUT transfer_function: transfer_function for layer
        %
        function add_layer(self, neurons, transfer_function)
            if ~self.initialized
                error('not initialized')
            else
                % get # of neurons in last layer
                inputs = size(self.layers{end}.w, 1);

                % initialize new layer with proper dimensions
                new_layer = layer;
                new_layer.initialize(neurons, inputs, transfer_function);
                self.layers{end+1} = new_layer;
            end
        end

        %
        % ADD_LAYERs adds consecutive layers to the network
        %
        % SYNOPSIS add_layers(self, neurons_list, transfer_function_list)
        %   the network must be initialized
        %   the number of neurons must match the number of transfer
        %       functions
        %
        % INPUT neurons_list: desired neurons in layer in vector format 
        %   [x ...] where x is a whole number
        % INPUT transfer_function: transfer_function for layer in vector
        %   format [x ...] where x is a function handle (ex. @logsig)
        %   
        function add_layers(self, neurons_list, transfer_function_list)
            if ~self.initialized
                error('not intialized')
            elseif length(neurons_list) ~= length(transfer_function_list)
                error('# of neurons != # of functions')
            end
            
            % add a layer
            for i = 1 : length(neurons_list)
                self.add_layer(neurons_list(i), transfer_function_list{i});
            end
        end
        
        %
        % FORWARD_PROPAGATION propagate p through the network
        %
        % SYNOPSIS [e] = forward_propagation(self, p, t)
        %   the network must be initialized
        %
        % INPUT p: pattern
        % INPUT t: target
        % OUTPUT e: error at final layer
        %
        function [e] = forward_propagation(self, p, t)
            % compute output of first layer
            a=self.layers{1}.activate(p);      
            
            % compute next layer using output of previous layer
            for m=2:length(self.layers)
                a=self.layers{m}.activate(a);
            end
            e=t-a;
        end
        
        %
        % BACKPROPAGATE_SENSITIVITIES backpropagate e through the network
        %
        % SYNOPSIS backpropagate_sensitivities(self, e)
        %
        % INPUT e: error at last layer
        %
        function backpropagate_sensitivities(self, e, p)
            % calculate last layer
            self.layers{end}.sensitivity_M(e, self.layers{end-1}.a);
            
            % then can calculate middle layers
            for m=length(self.layers)-1:-1:2
                 self.layers{m}.sensitivity_m(self.layers{m+1}.w, ...
                     self.layers{m+1}.s, self.layers{m-1}.a);
            end
            
            % calculate first layer
            self.layers{1}.sensitivity_m(self.layers{2}.w, ...
                     self.layers{2}.s, p);
        end
        
        %
        % UPDATE_LAYERS update layers of the network
        %
        % SYNOPSIS update_layers(self, p)
        %
        % INPUT p: pattern
        %
        function update_layers(self, p, batch_size)
            % m >= 2 so that m-1 != 0
            for m=length(self.layers):-1:2
                self.layers{m}.update(self.alpha, self.layers{m-1}.a, batch_size);
            end
            
            % update first layer
            self.layers{1}.update(self.alpha, p, batch_size);
        end
             
        %
        % TRAIN trains network using the given data set until the number of
        %   epochs has been reached
        %   uses the backpropagation algorithm
        %
        % SYNOPSIS train(self, epochs, training_set)
        %   the network must be initialized
        %   the number of epochs must be posative
        %   the batch_size must be posative and less than the length of the
        %       training_set
        %   the training set must be non-empty
        %
        % INPUT epochs: number of passes over the training set
        % INPUT batch_size: number of samples per batch
        % INPUT training_set: training data in the format { {p t} ... } 
        %   where p & t are column vectors
        %
        function [mse] = train(self, epochs, batch_size, training_set)
            samples = length(training_set);
            if ~self.initialized
                error('not initialized')
            elseif epochs < 1 
                error('no epochs')
            elseif batch_size < 1 || batch_size > samples
                error('batch size invalid')
            elseif isempty(training_set)
                error('no training set')
            end

            % plot data
            mse = zeros(1, epochs);
            
            % backpropagation algorithm
            for k=1:epochs
                for sample=1:batch_size:samples
                    for job=0:batch_size-1
                        %sample+job
                        p = training_set{sample+job}{1};
                        t = training_set{sample+job}{2};

                        % step 1: forward prop and get error
                        e = self.forward_propagation(p, t);

                        % step 2 & 3: back prop and set sensitivities
                        self.backpropagate_sensitivities(e, p);
                    end
                    % step 4: update w & b for each layer
                    self.update_layers(p, batch_size);
                end
                mse(k) = e' * e; 
            end
        end
        
        function n_correct = evaluate(self, test_set)
            n_correct=0;

            for sample=1:length(test_set)
                predicted=-1;                               % activated neuron in final layer
                predictions=zeros(size(test_set{1}{2}));    % vector to store values for final output
                max=0;                                      % used to track max 
                p=test_set{sample}{1};                      % input of test_set
                t=test_set{sample}{2};                      % expected output of test_set

                % Push test example through network
                a=layers{end}.a;

                % Iterate through output layer neurons and 
                % choose maximum of neurons as network's
                % prediction
                for i=1:length(a)
                 i
                 a(i,1)
                    % determine network prediction for test example
                    if a(i,1)>max
                      max=a(i,1);
                      predicted=i-1

                    end
                end
                 t
                 % network output
                 predictions(predicted+1,1)=1

                 % record network accuracy
                 n_correct = n_correct + isequal(predictions,t)
            end
        end
    end
end