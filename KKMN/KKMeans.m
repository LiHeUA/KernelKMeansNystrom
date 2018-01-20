function label = myKKMeans(Ke, k, l0)
% Kernel K-Means
n = size(Ke,1);

k = min(n,k);

if nargin==3
    label0 = l0;
else
    label0 = ceil(k*rand(n,1));
end


%% New Code
label = label0';
last = zeros(1,n);
iter = 0;
maxIter = 5;
while any(label ~= last) && iter<maxIter
    [~,~,last(:)] = unique(label);   % remove empty clusters
    E = sparse(last,1:n,1);
    E = bsxfun(@times,E,1./sum(E,2));
    T = E*Ke;
    Z = bsxfun(@minus,T,diag(T*E')/2);
    [dis, label] = max(Z,[],1);
    
    % eliminate empty class
    mx = max(dis);
    idxEmptyClu = setdiff(1:k,label);
    for i=1:length(idxEmptyClu)
        [~, idx] = min(dis);
        label(idx) = idxEmptyClu(i);
        dis(idx) = mx;
    end
    
    iter = iter+1;
end