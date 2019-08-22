function network = train_DBN(train_X, layers, iterations, training, verbose)
%TRAIN_DBN Trains an DBN to embede the data
%
%   network = train_DBN(train_X, layers, training)
%
% The used training technique is specified in
% training. Possible values are 'CD1' and 'PCD' (default = 'CD1').
%
%  Modified by Zahra Ghafoori @ unimelb, on 2016
% (C) Laurens van der Maaten
% Maastricht University, 2008

if nargin < 5,  verbose = 1; end;

if ~exist('training', 'var') || isempty(training)
    training = 'CD1';
end

% Pretrain the network
train_X = train_X(randperm(size(train_X,1)),:);
no_layers = length(layers);
network = cell(1, no_layers);
for i=1:no_layers
    if verbose
        % Print progress
        disp(['Training layer ' num2str(i) ' (size ' num2str(size(train_X, 2)) ...
            ' -> ' num2str(layers(i)) ')...']); 
    end;

    if i ~= no_layers

        % Train layer using binary units
        if strcmp(training, 'CD1')
            network{i} = train_rbm(train_X, layers(i), [], iterations);
        elseif strcmp(training, 'PCD')
            network{i} = train_rbm_pcd(train_X, layers(i), [], iterations);
        elseif strcmp(training, 'None')
            v = size(train_X, 2);
            network{i}.W = randn(v, layers(i)) * 0.1;
            network{i}.bias_upW = zeros(1, layers(i));
            network{i}.bias_downW = zeros(1, v);
        else
            error('Unknown training procedure.');
        end

        % Transform data using learned weights
        train_X = 1 ./ (1 + exp(-(bsxfun(@plus, train_X * network{i}.W, network{i}.bias_upW))));
    else

        if ~strcmp(training, 'None')
            % Train layer using Gaussian hidden units
            %network{i} = train_lin_rbm(train_X, layers(i), [], iterations);
            
            network{i} = train_rbm(train_X, layers(i), [], iterations);
        else
            v = size(train_X, 2);
            network{i}.W = randn(v, layers(i)) * 0.1;
            network{i}.bias_upW = zeros(1, layers(i));
            network{i}.bias_downW = zeros(1, v);
        end
    end
end
end