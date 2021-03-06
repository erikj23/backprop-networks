%% LAYER custom neural network layer implementation
% 
% PROPERTIES
%   w - weight matrix 
%   b - bias vector            
%   n - net input
%   a - output
%   s - sensitivity
%   f - transfer function
%   df - d/dn of transfer function
%   qw - batch accumulator for weight matrix
%   qb - batch accumulator for bias vector
classdef layer < handle
    properties(Access = 'private')        
        f;    % transfer function
        df;   % d/dn transfer function
        qw=0; % batch accumulator for weight matrix
        qb=0; % batch accumulator for bias vector
    end
    properties%(SetAccess = 'private')
        w;    % weight matrix
        b;    % bias vector
        n=0;  % net input
        a=0;  % output
        s=0;  % sensitivity
    end
    methods
        
        %
        % INITIALIZE initialize layer 
        %
        % SYNOPSIS initialize(self, neurons, inputs, transfer_function)
        %   sets w to be a [neurons x inputs] matrix with values between 
        %       [-1, 1]
        %   sets b to be a [neurons x 1] vector with values between [-1, 1]
        %   computes derivative of transfer function
        %
        % INPUT neurons: number of neurons in the layer
        % INPUT inputs: number of inputs for each neuron
        % INPUT transfer_function: transfer function
        %
        function initialize(self, neurons, inputs, transfer_function)            
            self.f=transfer_function;
            
            syms x;            
            if isequal(transfer_function, @softmax)
                % softmax layer
                self.df=matlabFunction(-log(x));
            else                
                % computing derivative of transfer function
                self.df=matlabFunction(diff(transfer_function(x)));
            end            
            
            % create initial matrix with values between [-1, 1]
            self.w = (rand(neurons, inputs) - 0.5) * 2;
            self.b = (rand(neurons, 1) - 0.5) * 2;
        end

        %
        % ACTIVATE applies transfer function to net input
        %
        % SYNOPSIS [a] = activate(self, p)
        %   stores a locally for later use
        %
        % INPUT p: pattern
        % OUTPUT a: output of transfer function
        %
        function [a] = activate(self, p)
            self.a = self.f(self.net_input(p));
            a = self.a;
        end

        %
        % NET_INPUT computes net input for neurons
        %
        % SYNOPSIS [n] = net_input(self, p)
        %   stores n locally for later use
        %
        % INPUT p: pattern
        % OUTPUT n: net input to transfer function
        %
        function [n] = net_input(self, p)
            
            self.n = self.w * p + self.b;
            n = self.n;
        end
        
        %
        % SENSITIVITY_M calculate sensitivity at last layer
        %
        % SYNOPSIS sensitivity_M(self, e)
        %   sets s locally for later use
        %
        % INPUT e: error at final layer (from forward propagation)
        % INPUT prev_a: output a(m-1) for batch computation
        %
        function sensitivity_M(self, t, prev_a)
            % computes jacobian matrix
            if isequal(self.f, @logsig)
                e = t - self.a;
                self.s = (-2) * self.df(self.n).* e;
            elseif isequal(self.f, @softmax)
                e = -sum(t .* log(self.a));
                self.s = (-2) * (t - self.a) .* e;
            elseif isequal(self.f, @purelin)
                e = t - self.a;
                self.s = (-2) * self.df() .* e;
            end
            
            % accumulate batch sensitivity
            self.qw = self.qw + self.s * prev_a';
            self.qb = self.qb + self.s;
        end
    
        %
        % SENSITIVITY_M calculate sensitivity at layer m
        %
        % SYNOPSIS sensitivity_m(self, next_w, next_s)
        %   sets s locally for later use
        %
        % INPUT next_w: weight matrix w(m+1)
        % INPUT next_s: sensitivity s(m+1)
        % INPUT prev_a: output a(m-1) for batch computation
        %
        function sensitivity_m(self, next_w, next_s, prev_a)
            self.s = self.df(self.n) .* next_w' * next_s;
            %self.s = (1 - self.a) .* self.a .* next_w' * next_s;
            
            % accumulate batch sensitivity
            self.qw = self.qw + self.s * prev_a';
            self.qb = self.qb + self.s;
        end
        
        %
        % UPDATE updates the current weight matrix and bias
        %
        % SYNOPSIS update(self, alpha, prev_a)
        %
        % INPUT alpha: learning rate
        % INPUT prev_a: output a(m-1)
        % INPUT batch_size: number of samples per batch
        %
        function update(self, alpha, prev_a, batch_size)
            if batch_size > 1
                self.w = self.w - (alpha * self.qw / batch_size);
                self.b = self.b - (alpha * self.qb / batch_size);
                
                % reset batch accumulators on update
                self.qw = 0;
                self.qb = 0;
            else
                self.w = self.w - (alpha * self.s * prev_a');
                self.b = self.b - (alpha * self.s);
            end
        end
    end
end