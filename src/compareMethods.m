function res = compareMethods(fileName,backPropMethod, gradMethod, options, layers, runNo)

%input/output files
f = who('-file', fileName);
directName = [pwd, '\comp_methods_results'];
mkdir(directName); %main directory to save results
mkdir([directName,'\',backPropMethod, '_', gradMethod]);
mkdir([directName,'\dae']);
mkdir([directName,'\PCA']);
mkdir([directName,'\t_SNE']);
mkdir([directName,'\RBM']);
mkdir([directName,'\1SVM']);

%common settings
dataSize = 10000; tSize = 0.8;
k = 'floor(0.05*trainSize)';
%anomalyPerc = 0.05;

%dae
iterations = 10;

%result
res = cell(size(f,1),6);

for i = 1:size(f,1)
    data = load(fileName,f{i});
    data = cell2mat(struct2cell(data));
    data = data(randperm(size(data,1)),:);
    for r = 1:runNo
        anomalyPerc = (0.05-0.01).*rand() + 0.01;
        [data1, trainSize] = pickData(data,anomalyPerc,dataSize,tSize);
        
        %%
        %1SVM
        [res{i,1}(r,7),res{i,1}(r,1:6)] = oneSVM(data1,trainSize,k,trainSize);
        res{i,1}(r,8) = anomalyPerc;
        acc_time = res{i,1};
        save([directName,'\1svm\',f{i}],'acc_time');
        fprintf('\n%s run %d;1svm %0.3f e %0.3f \n\n',f{i},r,res{i,1}(r,4),res{i,1}(r,7));
        
        %%
        %dae
        [res{i,2}(r,7),res{i,2}(r,1:6)] = dae(data1,trainSize,k,layers,iterations,trainSize);
        res{i,2}(r,8) = anomalyPerc;
        acc_time = res{i,2};
        save([directName,'\dae\',f{i}],'acc_time');
        fprintf('\n%s run %d;dae %0.3f e %0.3f \n\n',f{i},r,res{i,2}(r,4),res{i,2}(r,7));
        
        %%
        %pca
        [res{i,3}(r,7),res{i,3}(r,1:6)] = pcaTest(data1,trainSize,k,layers,trainSize);
        res{i,3}(r,8) = anomalyPerc;
        acc_time = res{i,3};
        save([directName,'\pca\',f{i}],'acc_time');
        fprintf('\n%s run %d;pca %0.3f e %0.3f \n\n',f{i},r,res{i,3}(r,4),res{i,3}(r,7));
        
        %%
        %t-SNE and RBM
        [eacc_time,accacc_time] = t_SNEandRBM(data1,trainSize,k,layers,trainSize);
        res{i,4}(r,7) = eacc_time(1); res{i,4}(r,1:6) = accacc_time(1,:);
        res{i,5}(r,7) = eacc_time(2); res{i,5}(r,1:6) = accacc_time(2,:);
        res{i,4}(r,8) = anomalyPerc; res{i,5}(r,8) = anomalyPerc;
        
        acc_time = res{i,4};
        save([directName,'\t_sne\',f{i}],'acc_time');
        fprintf('\n%s run %d;t_SNE %0.3f e %0.3f \n\n',f{i},r,res{i,4}(r,4),res{i,4}(r,7));
        
        acc_time = res{i,5};
        save([directName,'\RBM\',f{i}],'acc_time');
        fprintf('\n%s run %d;RBM %0.3f e %0.3f \n\n',f{i},r,res{i,5}(r,4),res{i,5}(r,7));
        
        %%
        %LN_SNE
        [res{i,6}(r,7),res{i,6}(r,1:6)] = LN_SNE(data1,trainSize,k,layers,options, backPropMethod, gradMethod,trainSize);
        res{i,6}(r,8) = anomalyPerc;
        acc_time = res{i,6};
        save([directName,'\',backPropMethod, '_', gradMethod,'\',f{i}],'acc_time');
        fprintf('\n%s run %d;LN_SNE %0.3f e %0.3f \n\n',f{i},r,res{i,6}(r,4),res{i,6}(r,7));
        
        %%
        if r > 1
            tmp = cell2mat(res(i,:));
            stats = [mean(tmp);std(tmp);min(tmp);max(tmp)];
            save([directName,'\',f{i}],'stats');
        end;
        save([directName,'\resAll'],'res');
    end
end
end

function [e,acc] = oneSVM(data,trainSize,k,bSize)
t = cputime;
[inds,gamma] = QMS2V1(data(1:trainSize,1:end-1),eval(k),1,bSize);
SVMop = ['-s 2 -h 0 -t 2 -g ',num2str(gamma),' -n 0.01'];
mdl = svmtrain(data(inds,end),data(inds,1:end-1),SVMop);
e = cputime-t;
[pLabel, ~, ~] = svmpredict(ones(size(data,1),1),data(:,1:end-1),mdl);
%train
[acc(2),acc(3),~] = calcContMatrix(data(1:trainSize,end),pLabel(1:trainSize));
acc(1) = trapz([0 acc(3) 1],[0 acc(2) 1]);
%test
[acc(5),acc(6),~] = calcContMatrix(data(trainSize+1:end,end),pLabel(trainSize+1:end));
acc(4) = trapz([0 acc(6) 1],[0 acc(5) 1]);
end
        
function [e,acc] = LN_SNE(data,trainSize,k,layers,options, backPropMethod, gradMethod,bSize)
acc = zeros(6,1);
t = cputime;
[network,~,inds] = train_par_tsneLN(data(1:trainSize,1:end-1),data(1:trainSize,end),...
    backPropMethod, gradMethod, layers, options, 'CD1');

data = [run_data_through_network(network, data(:,1:end-1)) data(:,end)];
maxim = max(data(1:trainSize,1:end-1)); minim = min(data(1:trainSize,1:end-1));
data(:,1:end-1) = rdivide(data(:,1:end-1) - ones(size(data,1),1)*minim,ones(size(data,1),1)*(maxim-minim));

if inds
    [~,gamma] = QMS2V1(data(inds,1:end-1),eval(k),0);
else
    [inds,gamma] = QMS2V1(data(1:trainSize,1:end-1),eval(k),1,trainSize);
end;
SVMop = ['-s 2 -h 0 -t 2 -g ',num2str(gamma),' -n 0.01'];
mdl = svmtrain(data(inds,end),data(inds,1:end-1),SVMop);
e = cputime-t;
[pLabel, ~, ~] = svmpredict(ones(size(data,1),1),data(:,1:end-1),mdl);
%train
[acc(2),acc(3),~] = calcContMatrix(data(1:trainSize,end),pLabel(1:trainSize));
acc(1) = trapz([0 acc(3) 1],[0 acc(2) 1]);
%test
[acc(5),acc(6),~] = calcContMatrix(data(trainSize+1:end,end),pLabel(trainSize+1:end));
acc(4) = trapz([0 acc(6) 1],[0 acc(5) 1]);
end

function [e,acc] = dae(data,trainSize,k,layers,iterations,bSize)
train_X = data(1:trainSize,1:end-1);
layers = eval(layers);
%layers = [floor(size(data,2)/2) layers];
t = cputime;
network = train_deep_autoencMe(data(1:trainSize,1:end-1), layers, 0,iterations,0,0);
data = [run_data_through_autoenc(network(1:length(layers)), data(:,1:end-1)) data(:,end)];
maxim = max(data(1:trainSize,1:end-1)); minim = min(data(1:trainSize,1:end-1));
data(:,1:end-1) = rdivide(data(:,1:end-1) - ones(size(data,1),1)*minim,ones(size(data,1),1)*(maxim-minim));
[inds,gamma] = QMS2V1(data(1:trainSize,1:end-1),eval(k),1,bSize);
SVMop = ['-s 2 -h 0 -t 2 -g ',num2str(gamma),' -n 0.01'];
mdl = svmtrain(data(inds,end),data(inds,1:end-1),SVMop);
e = cputime-t;
[pLabel, ~, ~] = svmpredict(ones(size(data,1),1),data(:,1:end-1),mdl);

%train
[acc(2),acc(3),~] = calcContMatrix(data(1:trainSize,end),pLabel(1:trainSize));
acc(1) = trapz([0 acc(3) 1],[0 acc(2) 1]);
%test
[acc(5),acc(6),~] = calcContMatrix(data(trainSize+1:end,end),pLabel(trainSize+1:end));
acc(4) = trapz([0 acc(6) 1],[0 acc(5) 1]);
end


function [e,acc] = pcaTest(data,trainSize,k,layers,bSize)
train_X = data(1:trainSize,1:end-1);
layers = eval(layers);
t = cputime;
mapping = cell(length(layers),1);
train_X = data(1:trainSize,1:end-1);
for mm = 1:length(layers)
    [~, mapping{mm}] = pca(train_X, layers(mm));
    train_X = train_X*mapping{mm}.M;
end;
for mm = 1:length(layers)
    data = [data(:,1:end-1)*mapping{mm}.M data(:,end)];
end;
maxim = max(data(1:trainSize,1:end-1)); minim = min(data(1:trainSize,1:end-1));
data(:,1:end-1) = rdivide(data(:,1:end-1) - ones(size(data,1),1)*minim,ones(size(data,1),1)*(maxim-minim));
[inds,gamma] = QMS2V1(data(1:trainSize,1:end-1),eval(k),1,bSize);
SVMop = ['-s 2 -h 0 -t 2 -g ',num2str(gamma),' -n 0.01'];
mdl = svmtrain(data(inds,end),data(inds,1:end-1),SVMop);
e = cputime-t;
[pLabel, ~, ~] = svmpredict(ones(size(data,1),1),data(:,1:end-1),mdl);

%train
[acc(2),acc(3),~] = calcContMatrix(data(1:trainSize,end),pLabel(1:trainSize));
acc(1) = trapz([0 acc(3) 1],[0 acc(2) 1]);
%test
[acc(5),acc(6),~] = calcContMatrix(data(trainSize+1:end,end),pLabel(trainSize+1:end));
acc(4) = trapz([0 acc(6) 1],[0 acc(5) 1]);
end

function [e,acc] = t_SNEandRBM(data,trainSize,k,layers,bSize)
train_X = data(1:trainSize,1:end-1);
% t = cputime;
% [network,~,networkRBM,e(2),inds] = train_par_tsneV0(data(1:trainSize,1:end-1),....
%     data(1:trainSize,end),'tsne_grad',eval(layers),[],'CD1');
% 
% data1 = [run_data_through_network(network, data(:,1:end-1)) data(:,end)];
% maxim = max(data1(1:trainSize,1:end-1)); minim = min(data1(1:trainSize,1:end-1));
% data1(:,1:end-1) = rdivide(data1(:,1:end-1) - ones(size(data1,1),1)*minim,ones(size(data1,1),1)*(maxim-minim));
% 
% [~,gamma] = QMS2V1(data1(inds,1:end-1),eval(k),0);
% %[inds,gamma] = QMS2V1(data1(1:trainSize,1:end-1),eval(k),1,bSize);
% SVMop = ['-s 2 -h 0 -t 2 -g ',num2str(gamma),' -n 0.01'];
% mdl = svmtrain(data1(inds,end),data1(inds,1:end-1),SVMop);
% e(1) = cputime-t;

t = cputime;
[network,~,networkRBM,e(2)] = train_par_tsneV1(data(1:trainSize,1:end-1),....
    data(1:trainSize,end),'tsne_grad',eval(layers),[],'CD1');

data1 = [run_data_through_network(network, data(:,1:end-1)) data(:,end)];
maxim = max(data1(1:trainSize,1:end-1)); minim = min(data1(1:trainSize,1:end-1));
data1(:,1:end-1) = rdivide(data1(:,1:end-1) - ones(size(data1,1),1)*minim,ones(size(data1,1),1)*(maxim-minim));

[inds,gamma] = QMS2V1(data1(1:trainSize,1:end-1),eval(k),1,bSize);
SVMop = ['-s 2 -h 0 -t 2 -g ',num2str(gamma),' -n 0.01'];
mdl = svmtrain(data1(inds,end),data1(inds,1:end-1),SVMop);
e(1) = cputime-t;

[pLabel, ~, ~] = svmpredict(ones(size(data1,1),1),data1(:,1:end-1),mdl);

%train
[acc(1,2),acc(1,3),~] = calcContMatrix(data1(1:trainSize,end),pLabel(1:trainSize));
acc(1,1) = trapz([0 acc(1,3) 1],[0 acc(1,2) 1]);
%test
[acc(1,5),acc(1,6),~] = calcContMatrix(data1(trainSize+1:end,end),pLabel(trainSize+1:end));
acc(1,4) = trapz([0 acc(1,6) 1],[0 acc(1,5) 1]);

%for RBM
t = cputime;
data1 = [run_data_through_network(networkRBM, data(:,1:end-1)) data(:,end)];
maxim = max(data1(1:trainSize,1:end-1)); minim = min(data1(1:trainSize,1:end-1));
data1(:,1:end-1) = rdivide(data1(:,1:end-1) - ones(size(data1,1),1)*minim,ones(size(data1,1),1)*(maxim-minim));

%[~,gamma] = QMS2V1(data1(inds,1:end-1),eval(k),0);
[inds,gamma] = QMS2V1(data1(1:trainSize,1:end-1),eval(k),1,bSize);
SVMop = ['-s 2 -h 0 -t 2 -g ',num2str(gamma),' -n 0.01'];
mdl = svmtrain(data1(inds,end),data1(inds,1:end-1),SVMop);
e(2) = e(2)+cputime-t;
[pLabel, ~, ~] = svmpredict(ones(size(data1,1),1),data1(:,1:end-1),mdl);

%train
[acc(2,2),acc(2,3),~] = calcContMatrix(data1(1:trainSize,end),pLabel(1:trainSize));
acc(2,1) = trapz([0 acc(2,3) 1],[0 acc(2,2) 1]);
%test
[acc(2,5),acc(2,6),~] = calcContMatrix(data1(trainSize+1:end,end),pLabel(trainSize+1:end));
acc(2,4) = trapz([0 acc(2,6) 1],[0 acc(2,5) 1]);
end