function [v] = dec2vec(dec)
    switch dec
        case 0
            v = [1 0 0 0 0 0 0 0 0 0]';
        case 1
            v = [0 1 0 0 0 0 0 0 0 0]';
        case 2
            v = [0 0 1 0 0 0 0 0 0 0]';
        case 3
            v = [0 0 0 1 0 0 0 0 0 0]';
        case 4
            v = [0 0 0 0 1 0 0 0 0 0]';
        case 5
            v = [0 0 0 0 0 1 0 0 0 0]';
        case 6
            v = [0 0 0 0 0 0 1 0 0 0]';
        case 7
            v = [0 0 0 0 0 0 0 1 0 0]';
        case 8
            v = [0 0 0 0 0 0 0 0 1 0]';
        case 9
            v = [0 0 0 0 0 0 0 0 0 1]';
        otherwise
            error('invalid input')
    end
end