function [approxK, vec, lambda_z, label] = KernelKMeanApproximation(K, c)
% Main function of kernel k-means sampling for Nystrom approximation in [1]
% In [1] we show that, kernel k-means minimizes the apprx. upper bound of
% Nystrom approximation. 
% 
% [1] Li He and Hong Zhang, Kernel K-means Sampling for Nystrom 
% Approximation, to appear in IEEE Transactions on Image Processing.
%
% Input: 
%       K           n*n         Kernel matrix of all n data points
%       c           scalar      Sample size
% Output:
%       approxK     n*n         Approximated K
%       vec         n*c         Approximated eigenvectors of K
%       lambda_z    c*1         Approximated eigenvalues of K
%       labels      1*n         Kernel k-means labels
%
% heli@gdut.edu.cn


% number of data
numData = size(K,1);
% training size
szTrain = c;

%% 1. Kernel K Means Sampling
label = KKMeans(K,szTrain);

idx = cell(1,szTrain);
sz = zeros(1,szTrain);
len = zeros(1,szTrain);
for i=1:szTrain
    idx{i} = find(label==i);
    sz(i) = length(idx{i});
    len(i) = sqrt(sum(sum(K(idx{i},idx{i}))));
end

%% 2. Ksz
Ksz = zeros(szTrain, szTrain);
for i=1:szTrain
    for j=i:szTrain
        idxp = idx{i};
        idxq = idx{j};
        Kpq = K(idxp,idxq);
        Ksz(i,j) = sum(sum(Kpq))/len(i)/len(j);
    end
end
Ksz = Ksz+Ksz'-diag(diag(Ksz));
Ksz = bsxfun(@times,Ksz,sz);

%% 3. Compute K_NZ
Knz = zeros(numData, szTrain);
for j=1:szTrain
    idxq = idx{j};
    Kpq = K(:,idxq);
    Knz(:,j) = sum(Kpq,2)/len(j);
end

%% 4. Approximate {v_z, lambda_z} and tilde{K}
[vec_sz, val_sz] = eig(Ksz);
% lambda_z = val_sz;
[lambda_z, idxv] = sort(diag(val_sz),'descend');
tmp = lambda_z>1e-5;
lambda_z = lambda_z(tmp);
vec_sz = vec_sz(:,idxv(tmp));

%% 5. Extension
Knz = bsxfun(@times,Knz,sz);
nm = bsxfun(@times,vec_sz.^2,sz');
v_z = bsxfun(@times,vec_sz,1./sqrt(sum(nm)));


vec = bsxfun(@times,Knz*v_z,1./lambda_z');

% approxK = vec*lambda_z*vec';
approxK = bsxfun(@times,vec,lambda_z');
approxK = approxK*vec';