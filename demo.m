function demo
% Demo of kernel k-means sampling for Nystrom approximation in [1].
% In [1] we show that, kernel k-means minimizes the apprx. upper bound of
% Nystrom approximation. 
%
% [1] Li He and Hong Zhang, Kernel K-means Sampling for Nystrom 
% Approximation, to appear in IEEE Transactions on Image Processing.
%
% heli@gdut.edu.cn

clc
close all

addpath('./KKMN'); % Our main function

%% 0. Initialization

% select one dataset to run your experiment
load ./svmguide2.mat; % data, labels
% load ./svmguide4.mat;
% load ./LiverDisorders.mat;
% load ./Ionosphere.mat;

% Distance matrix of input data
dis = pdist2(data,data);
% Set sigma = avg. of dis.
sigma = mean(dis(:));

% Ground Truth K
K = exp(-dis.^2/sigma^2);

%% 1: Kernel K-Means Nystrom on Raw Data
ratioList = [.1:.1:.5];
cList = floor(ratioList*size(data,1));
for i=1:length(cList)
    c = cList(i);
    apprxK_KKM = KernelKMeanApproximation(K, c);
    err(i) = norm(K-apprxK_KKM,'fro')/norm(K,'fro'); % relative error
end

%% 2. Show Result
plot(ratioList,err,'r*-');
xlabel('Sample size ratio')
ylabel('Relative approximation error')