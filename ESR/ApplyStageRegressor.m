function [ delta_y ] = ApplyStageRegressor( I,S,fern,trans_matrix,la,delta_la,Nfp,P,F,K )
%ApplyStageRegressor Apply regressor on test image and shape
%   此处显示详细说明
N=size(I,1);
ShapeIndexedFeatures=zeros(N,P,'single');
DiffFeatures=zeros(N,F,'single');
delta_la=reshape(delta_la,P,2);
delta_y=zeros(N,2*Nfp);
encoder=[1 2 4 8 16];
for i=1:N
    Stmp=reshape(S{i,1},[74,2]);
    tmp=Stmp(la,:)+delta_la/(trans_matrix{i,1}*trans_matrix{i,2});
    tmp(tmp>250)=250;
    tmp(tmp<1)=1;
    tmp=int32(tmp);
    idx=250*(tmp(:,1)-1)+tmp(:,2);
    ShapeIndexedFeatures(i,:)=I(i,idx);
end

for t=1:K
    I_idx=fern{t}{1};
    theta=fern{t}{2};
    for i=1:F
   DiffFeatures(:,i)=ShapeIndexedFeatures(:,I_idx(i,1))-ShapeIndexedFeatures(:,I_idx(i,2));
    end
    regressor=fern{t}{3};
   partition=bsxfun(@gt,single(DiffFeatures),theta);
   bin_num=sum(bsxfun(@times,partition,encoder),2)+1; 
    for i=1:N
        delta_y(i,:)=delta_y(i,:)+regressor(bin_num(i),:);
    end
end
end 