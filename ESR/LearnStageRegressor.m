function [ delta_s,fern,la,delta_la] = LearnStageRegressor( Y,I,S_t,M_t,N_aug,Nfp,P,lambda,F,K )
%LearnStageRegressor Learn PrimitiveRegressor
%   此处显示详细说明
[ShapeIndexedFeatures,la,delta_la]=GenerateShapeIndexedFeatures(I,S_t,M_t,N_aug,Nfp,P,lambda); 
cov_feat=cov(ShapeIndexedFeatures);
N=size(I,1);
num=N*N_aug;
delta_s=zeros(num,2*Nfp);
encoder=[1 2 4 8 16];
fern=cell(1,K);
%cov_feat=cov(ShapeIndexedFeatures);

%F=5;
%K=1;
for t=1:K
    fern{t}=cell(3,1);
    [fern{t}{1},DiffFeatures] = CorrelationBasedFeature( Y,ShapeIndexedFeatures,cov_feat,P,F);
    %fern{1,t}{1,1}
    %norm(Y,'fro')
    c=single(max(abs(DiffFeatures(:))));
    theta=0.4*c.*(rand(1,F)-0.5);
    fern{t}{2}=theta;
    partition=bsxfun(@gt,single(DiffFeatures),theta);
    bin_num=sum(bsxfun(@times,partition,encoder),2)+1;
    %size(partition)
    omega_b=zeros(2^F,1);
    y_b=zeros(2^F,2*Nfp,'single');
    for i=1:num
        omega_b(bin_num(i))=omega_b(bin_num(i))+1;
        y_b(bin_num(i),:)=y_b(bin_num(i),:)+Y(i,:);
    end
    omega_b=omega_b+1000;
    regressor=bsxfun(@rdivide,y_b,omega_b);
    fern{t}{3}=regressor;
    for n=1:num
        Y(n,:)=Y(n,:)-regressor(bin_num(n),:);
        delta_s(n,:)=delta_s(n,:)+regressor(bin_num(n),:);
    end
end
%end

