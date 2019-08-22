function [network, err, Ind] = train_par_tsneLN(train_X, train_labels, backPropMethod, gradMethod, layers, options, training)
%TRAIN_PAR_TSNE Trains a parametric t-SNE embedding
%
%   [network, err] = train_par_tsne(train_X, train_labels, test_X,
%   test_labels, layers, training)
%
% Trains up a parametric t-SNE embedding with the structure that is
% specified in layers. The used training technique is specified in
% training. Possible values are 'CD1' and 'PCD' (default = 'CD1').
%
% Modified by Zahra Ghafoori 2016
% (C) Laurens van der Maaten
% Maastricht University, 2008


if ~exist('training', 'var') || isempty(training)
    training = 'CD1';
end

iterations = 1;
% [train_X,inds,ic] = unique(train_X,'rows');
% train_labels = train_labels(inds);

layers = eval(layers);%[floor(size(train_X,2)/2) floor(size(train_X,2)/4)];
% while layers(end) > 10
%     layers = [layers floor(layers(end)/2)];
% end;

% Pretrain the network
network = train_DBN(train_X,layers,iterations,training,options(4));

% compose string used to call function
argstr = [backPropMethod, '(network,train_X, train_labels, gradMethod, options)'];                     

% Perform backpropagation of the network using t-SNE gradient
network = eval(argstr);