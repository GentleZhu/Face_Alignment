function [ I_idx,DiffFeatures ] = CorrelationBasedFeature( Y,feat,cov_feat,P,F )
%CorrelationBasedFeature select F candidate feature
%   此处显示详细说明
[N,~]=size(Y);
Yproj=Y*randn(148,1);

Pearson_corr=zeros(P,P,'single');
cov_y_feat=zeros(1,P);
for i=1:P
    tt=cov(Yproj,feat(:,i));
    cov_y_feat(i)=tt(1,2);
end
tt=std(Yproj);
for i=1:P
    for j=i+1:P
        Pearson_corr(i,j)=(cov_y_feat(i)-cov_y_feat(j))/sqrt(tt*(cov_feat(i,i)+cov_feat(j,j)-2*cov_feat(i,j)));
    end
end
I_idx=zeros(F,2);
DiffFeatures=zeros(N,F,'single');
Pearson_corr=abs(Pearson_corr);
for i=1:F
    [~,I] = max(Pearson_corr(:));
    Pearson_corr(I)=0;
    [I_idx(i,1),I_idx(i,2)] = ind2sub([P,P],I);
    DiffFeatures(:,i)=feat(:,I_idx(i,1))-feat(:,I_idx(i,2));
end

end

