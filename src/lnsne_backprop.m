function network = lnsne_backprop(network, train_X, train_labels, gradMethod, options)
%LN-SNE_BACKPROP Perform fine-tuning of network using the LN-SNE gradient
%Inputs
%   network: RBM network initialised by CD to be fine-tunned using LN-SNE
%   train_X: input data
%   train_labels: labels for step-by-step visualisation (not used for training)
%   options: a vector that specifies values for settings:
%       max_iter, perplexity, verbos (for debugging),
%       plotF(for visualising step by step), btSize, lineSiteration
%Output: fine-tunned network
 
% Revised vesrion of tsne_backprop
%@Zahra Ghafoori, 2016


if or(~exist('options', 'var'), isempty(options))
    max_iter = 1; perplexity = 40; 
    verbos = 0; plotF = 0;
    btSize = 1000; lineSiteration = 3;
else
    max_iter = options(1); perplexity = options(2); 
    plotF = options(3); verbos = options(4); 
    btSize = options(5); lineSiteration = options(6);
end

% Initialize some variables
n = size(train_X, 1);
batch_size = min([btSize n]);
ind = randperm(n);

% Precompute joint probabilities for all batches
if verbos, disp('Precomputing P-values...'); end;
curX = cell(floor(n ./ batch_size), 1);
P = cell(floor(n ./ batch_size), 1);

i = 1;
k = floor(0.05*batch_size);
for batch=1:batch_size:n
    if batch + batch_size - 1 <= n
        ix = ind(batch:min([batch + batch_size - 1 n]));
        curX{i} = double(train_X(ix,:));
        [beta,D] = x2p1(curX{i}, perplexity, 1e-5,verbos);
        P{i} = exp(-D * beta);
               
        d = sort(D,2);
        d = mean(d(:,2:k),2);
        Q2 = median(d);
        Q3 = median(d((d>=Q2)));
        inds = find(d < (Q3+1.35*std(d)));
        l = -ones(batch_size,1);
        l(inds) = 1;
        P{i}(inds(sum(l==-1):end),:) = [];
        P{i}(:,inds(sum(l==-1):end)) = [];
        curX{i}(inds(sum(l==-1):end),:) = [];
        
        P{i}(1:size(P{i},1)+1:end) = 0;
        P{i} = P{i} ./ sum(P{i}(:));
        P{i} = max(P{i}, eps);
        i = i + 1;
    end
end

if plotF
    mapped_train_X = run_data_through_network(network, train_X);
    scatter(mapped_train_X(:,1), mapped_train_X(:,2), 9, train_labels);
    title('Embedding of train data');
    drawnow
end;

% Run the optimization
for iter=1:max_iter
    
    % Run for all mini-batches
    if verbos, disp(['Iteration ' num2str(iter) '...']); end;
    b = 1;
    for batch=1:batch_size:n
        if batch + batch_size - 1 <= n
            
            % Construct current solution
            x = [];
            for i=1:length(network)
                x = [x; network{i}.W(:); network{i}.bias_upW(:)];
            end
            
            % Perform conjugate gradient using three linesearches
            [x,~,~,ls_failed] = minimizeF(x, gradMethod, lineSiteration, verbos, curX{b}, P{b}, network);
            b = b + 1;
            
            % Store new solution
            ii = 1;
            for i=1:length(network)
                network{i}.W = reshape(x(ii:ii - 1 + numel(network{i}.W)), size(network{i}.W));
                ii = ii + numel(network{i}.W);
                network{i}.bias_upW = reshape(x(ii:ii - 1 + numel(network{i}.bias_upW)), size(network{i}.bias_upW));
                ii = ii + numel(network{i}.bias_upW);
            end
        end
        if ls_failed, break; end;
    end
    
    if plotF
        mapped_train_X = run_data_through_network(network, train_X);
        scatter(mapped_train_X(:,1), mapped_train_X(:,2), 9, train_labels);
        title('Embedding of train data');
        drawnow
    end;
end
end