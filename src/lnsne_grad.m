function [C, dC] = lnsne_grad(x, X, P, network)
%TSNE_GRAD Computes the t-SNE gradient w.r.t. weights in a network
%
%   [C, dC] = tsne_grad(x, X, P, network, v)
%
% Computes the t-SNE gradient w.r.t. weights in a neural network. The
% weights are encoded in x. The data is specified in X, whereas the 
% corresponding P-values are provided in P. Information on the number and 
% size of the layers is obtained from network. The degrees of freedom of
% the Student-t distribution that is employed can be specified through v.
%
%
% (C) Laurens van der Maaten
% Maastricht University, 2008


    % Initalize some variables
    n = size(X, 1);
    no_layers = length(network);

    % Update the network to store the new solution
    ii = 1;
    for i=1:length(network)
        network{i}.W = reshape(x(ii:ii - 1 + numel(network{i}.W)), size(network{i}.W)); 
        ii = ii + numel(network{i}.W);
        network{i}.bias_upW = reshape(x(ii:ii - 1 + numel(network{i}.bias_upW)), size(network{i}.bias_upW));
        ii = ii + numel(network{i}.bias_upW);
    end
    
    % Run the data through the network
    activations = cell(1, no_layers + 1);
    activations{1} = [X ones(n, 1)];
    for i=1:no_layers - 1
        activations{i + 1} = [1 ./ (1 + exp(-(activations{i} * [network{i}.W; network{i}.bias_upW]))) ones(n, 1)];
    end
    activations{end} = activations{end - 1} * [network{end}.W; network{end}.bias_upW]; 
    
    % Compute the Q-values
    num = pdist2(activations{end}, activations{end});
    mu = 0; sigma = 100; reloc = 100; relocY = 0;
    
    Q = lognpdf(num+reloc,mu,sigma)+relocY;
    Q(1:n+1:end) = 0;  % set diagonal to zero
    Q = Q ./ sum(Q(:)); % normalize to get probabilitiess
    Q = max(Q, eps);
    num = max(num,eps);
    
    C = sum(sum(P .* log((P + eps) ./ (Q + eps))));
    
    %Compute the derivatives w.r.t. the map coordinates (= errors)
    Ix = zeros(size(activations{end}));
    tmp = sigma^2+(log(num+reloc)-mu);
    stiffnesses = -tmp./(sigma^2.*num.*(num+reloc));
    tmp = 2*(P - Q);
    stiffnesses = -stiffnesses.*tmp;
    for i=1:n
        ys = bsxfun(@minus, activations{end}(i,:), activations{end});
        Ix(i,:) =  sum(bsxfun(@times, ys, stiffnesses(:,i)), 1);
    end;

    
    % Compute gradients 
    dW = cell(1, no_layers);
    db = cell(1, no_layers);
    for i=no_layers:-1:1

        % Compute update
        delta = activations{i}' * Ix;
        dW{i} = delta(1:end - 1,:);
        db{i} = delta(end,:);

        % Backpropagate error
        if i > 1
            Ix = (Ix * [network{i}.W; network{i}.bias_upW]') .* activations{i} .* (1 - activations{i});
            Ix = Ix(:,1:end - 1);
        end
    end

    % Convert gradient information
    dC = repmat(0, [numel(x) 1]);
    ii = 1;
    for i=1:no_layers
        dC(ii:ii - 1 + numel(dW{i})) = dW{i}(:); 
        ii = ii + numel(dW{i});
        dC(ii:ii - 1 + numel(db{i})) = db{i}(:); 
        ii = ii + numel(db{i});
    end