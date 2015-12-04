function [ShapeIndexedFeatures,la,delta_la] = GenerateShapeIndexedFeatures( I,S,trans_matrix,N_aug,Nfp,P,k )
%ESRTraining Explicit Shape Regression Training function 
%   此处显示详细说明
%load 'LFW.mat'
tic;
[N,~,~]=size(I);
%P=400;
%Nfp=74;
la=randi([1 Nfp],P,1);
%k=10;
delta_la=k/1000*randi([-1000 1000],P,2);
ShapeIndexedFeatures=zeros(N*N_aug,P,'single');
%tmp=zeros(400,2);
%tmp=[];
for i=1:N
    for k=1:N_aug
    %for a=1:P
        %index_l=la;
        idx=(i-1)*N_aug+k;
        Stmp=reshape(S{i,k},[74,2]);
        tmp=Stmp(la,:)+delta_la/(trans_matrix{idx,1}*trans_matrix{idx,2});
        tmp(tmp>250)=250;
        tmp(tmp<1)=1;
        tmp=int32(tmp);
        imgidx=250*(tmp(:,1)-1)+tmp(:,2);
        ShapeIndexedFeatures(idx,:)=I(i,imgidx);%这个地方纠结过
    end
    %end
end
%[~,sortIndex] = sort(abs(Pearson_corr(:)),'descend','omitnan');
toc
%ShapeIndexedPixelDiffFeatures=zeros(7258,P*P,'int16'); 
%
%cov_feat=cov(ShapeIndexedFeatures);
%cov_y_feat=cov(Yproj,ShapeIndexedFeatures);
%Pearson_col=zeros(1,P*P);
%for i=1:P
%    Pearson_col(:,(i-1)*P+1:i*P)=repmat(ShapeIndexedFeatures(:,i),1,P)-ShapeIndexedFeatures;
%end
%Pearson_col=corr(Yproj,single(ShapeIndexedPixelDiffFeatures));
%[~,idx]=sort(Pearson_col,'descend');
%candidate=idx(P+1:P+5);
%selected=ShapeIndexedPixelDiffFeatures(:,candidate);
%toc
end
%ShapeIndexedFeatures=int8(ShapeIndexedFeatures-128);