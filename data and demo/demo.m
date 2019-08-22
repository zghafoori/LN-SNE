%loading data (should be scaled in the range [0,1] beforehand
load('httpNslkdd');
%addpath for the method
addpath([pwd,'\LN-SNE']);


layers = '2';
options = [1 40 0 0 1000 3]; % for default

network = train_par_tsneLN(data(:,1:end-1),data(:,end),...
    'lnsne_backprop', 'lnsne_grad', layers, options, 'CD1');

mapped_data = [run_data_through_network(network, data(:,1:end-1)),data(:,end)];

%visualising the projected data
scatter(mapped_data(mapped_data(:,end)==1,1),mapped_data(mapped_data(:,end)==1,2),'.b')
hold on
scatter(mapped_data(mapped_data(:,end)==-1,1),mapped_data(mapped_data(:,end)==-1,2),'+r')
