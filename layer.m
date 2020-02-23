%% LAYER custom neural network layer implementation
% 
% PROPERTIES
%   w - weight matrix 
%   b - bias vector            
%   f - transfer function
%   df - d/dn of transfer function
%   n - net input
%   a - output
%   s - sensitivity
classdef layer < handle
    properties%(SetAccess = 'private')        
        w;  % weight matrix 
        b;  % bias vector            
        f;  % transfer function
        df; % d/dn transfer function
        n;  % net input
        a;  % output
        s;  % sensitivity
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
            syms x;
            self.f = transfer_function;
            
            % computing derivative of transfer function
            self.df = matlabFunction(diff(transfer_function(x)));
            
            % create initial matrix with values between [-1, 1]
            self.w = (rand(neurons, inputs)-0.5)*2;
            self.b = (rand(neurons, 1)-0.5)*2;
            
            % initialize net input and output and sensitivity
            self.n = 0;
            self.a = 0;
            self.s = 0;
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
            self.a = self.a+self.f(self.net_input(p));
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
            self.n = self.w*p+self.b;
            n = self.n;
        end
        
        %
        % SENSITIVITY_M calculate sensitivity at last layer
        %
        % SYNOPSIS sensitivity_M(self, e)
        %   sets s locally for later use
        %
        % INPUT e: error at final layer (from forward propagation)
        %
        function sensitivity_M(self, e)
            % chapter 11 example uses purelin, so d/dn = 1, parameters = 0
            % checking number of parameters with nargin
            % s = old s + new s, call to update sets to 0, for batch size 1
            %   call is equivalent to s = 0 + new s
            if nargin(self.df)
                self.s = self.s+(-2)*self.df(self.n).*e;
            else
                self.s = self.s+(-2)*self.df().*e;
            end
        end
    
        %
        % SENSITIVITY_M calculate sensitivity at layer m
        %
        % SYNOPSIS sensitivity_m(self, next_w, next_s)
        %   sets s locally for later use
        %
        % INPUT next_w: weight matrix w(m+1)
        % INPUT next_s: sensitivity s(m+1)
        %
        function sensitivity_m(self, next_w, next_s)
            % s = old s + new s, call to update sets to 0, for batch size 1
            %   call is equivalent to s = 0 + new s
            self.s = self.s+self.df(self.n).*next_w'*next_s;
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
            self.w = self.w - (alpha*(self.s*prev_a')/batch_size);
            self.b = self.b - (alpha*(self.s/batch_size));
            
            % reset output & sensitivity on batch update
            self.a = 0;
            self.s = 0;
        end
    end
end