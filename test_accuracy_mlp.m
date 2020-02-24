function error_rates = test_accuracy_mlp(inputs,n_pixels, network)
    
    % One accuracy rate per pattern
    accuracy_rates = zeros(length(inputs(1,:)),1);

    n_correct = 0;
    
    for i=1:length(inputs(1,:))-1
        
        % Test accuracy of each input
        for j=1:10

            % Add noise to test patterns
            corrupted = addNoise(inputs(:,k),n_pixels);

            % Test network
            a = network.forward_propagation(corrupted, inputs),;
            n_errors = n_errors + ~isequal(pattern, inputs(:,k));
        end
        
        % Calc error rate and reset for next digit
        error_rates(i,1) = (n_errors/(10*n_digits))*100;
        n_errors=0;
    end
end