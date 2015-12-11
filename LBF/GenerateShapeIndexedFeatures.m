function [ShapeIndexedPixelDiffFeatures] = GenerateShapeIndexedFeatures( I,S,trans_matrix,N_aug,Nfp,P,r)
%ESRTraining Explicit Shape Regression Training function 
%   此处显示详细说明
%load 'LFW.mat'
tic;
[N,~,~]=size(I);
%P=400;
%Nfp=74;
%la=randi([1 Nfp],P,1);
%k=10;
%P=(2*k+1)*(2*k+1);

ShapeIndexedFeatures=zeros(Nfp,N,P,'single');
%ShapeIndexedFeatures=[];
%tmp=zeros(400,2);
%tmp=[];
I=single(I);

pts_x=zeros(Nfp,P,'single');
pts_y=zeros(Nfp,P,'single');

for i=1:Nfp
    Ns = round(1.28*P + 2.5*sqrt(P) + 100); % 4/pi = 1.2732
    X = rand(Ns,1)*(2*r) - r;
    Y = rand(Ns,1)*(2*r) - r;
    Ix = find(sqrt(X.^2 + Y.^2)<=r);
    pts_x(i,:) = X(Ix(1:P));
    pts_y(i,:)= Y(Ix(1:P)); 
end
% trans=zeros(2,2,N,'single');
% 
% for i=1:N
%     trans(:,:,i)=trans_matrix{i,1}*trans_matrix{i,2};
% end
%pts_x(pts_x>250)=250;
%pts_y(pts_y<1)=1;
for i=1:N
    temp=reshape(S{i,1},[Nfp,2]);
    image=reshape(I(i,:,:),[250 250]);
    for j=1:Nfp
        %idx=bsxfun(@plus,[pts_x(i,:);pts_y(i,:)]',temp(j,:));
        %size(trans_matrix(:,1)*trans_matrix(i,2))
        %bsxfun(@rdivide,[pts_x(j,:);pts_y(j,:)]',trans);
        
        tmp=bsxfun(@plus,[pts_x(j,:);pts_y(j,:)]'/(trans_matrix{i,1}*trans_matrix{i,2}),temp(j,:));
        tmp(tmp>250)=250;
        tmp(tmp<1)=1;
        %tt=[tt;tmp];
        ShapeIndexedFeatures(j,i,:)=interp2(image,tmp(:,1),tmp(:,2));
        %tmp=int32(tmp);
        %imgidx=250*(tmp(:,1)-1)+tmp(:,2);
        %size(imgidx)
        %I(i,imgidx);
        %ShapeIndexedFeatures(j,i,:)=I(i,imgidx);
        %ShapeIndexedFeatures=[ShapeIndexedFeatures;tmp];
    end
end
ShapeIndexedPixelDiffFeatures=zeros(Nfp,N,P*P,'single');
for l=1:Nfp
for i=1:P
    ShapeIndexedPixelDiffFeatures(l,:,(i-1)*P+1:i*P)=repmat(ShapeIndexedFeatures(l,:,i),1,1,P)-ShapeIndexedFeatures(l,:,:);
end
end



% for i=1:N
%     dx=uint8(reshape(S{i,1},[Nfp 2]));
%     index(:,1)=dx(:,1)-k;
%     index(:,2)=dx(:,1)+k;
%     index(:,3)=dx(:,2)-k;
%     index(:,4)=dx(:,2)+k;
%     %index(index<1)=1;
%     %index(index>250)=250;
%     for j=1:Nfp
%     %size(I(i,index(j,1):index(j,2),index(j,3):index(j,4)))
%     %if index(j,1)==index(j,2)
%     ShapeIndexedFeatures(i,j,:)=reshape(I(i,index(j,3):index(j,4),index(j,1):index(j,2)),[1 P]);
%     end
% end
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