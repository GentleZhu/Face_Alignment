function [fern,la,delta_la,S_t,out] = ESRTraining( I,S0,init_scope,meanshape,N_aug,T,init_set )
%function    [S_t] = ESRTraining( I,S0,init_scope,meanshape,N_aug,T,init_set )
%ESRTraining Explicit Shape Regression Training function 
%   此处显示详细说明
N=size(I,1);
%S_t=cell(T,1);
%M_t=cell(T,1);

tic;
%S_t(1)={Initialization(N,N_aug,init_scope,init_set)};
S_t=Initialization(N,N_aug,init_scope,init_set);
%S_t(1)=init_set;
toc
trans_matrix=cell(N*N_aug,3);
Y=zeros(N*N_aug,148,'single');
S0=single(S0);
%F=5;
K=500;
out=zeros(T,1);
%encoder=[1 2 4 8 16];
la=zeros(400,T,'single');
delta_la=zeros(400,2,T,'single');
fern=cell(T,K);
for t=1:T
    tic;
    for n=1:N
        for k=1:N_aug
        idx=(n-1)*N_aug+k;
        %[~,Ytmp,trans] = procrustes(meanshape,reshape(S_t{t}{n,k},[74,2]));
        %size(S_t)
        [~,Ytmp,trans] = procrustes(meanshape,reshape(S_t{n,k},[74,2]));
        trans_matrix(idx,:)=struct2cell(trans)';
        Y(idx,:)=reshape(trans_matrix{idx,2}*reshape(S0(n,:,:),[74,2])*trans_matrix{idx,1}+trans_matrix{idx,3}-Ytmp,[1,148]);
        end
    end
    out(t)=norm(Y,'fro');
    toc
    %M_t(t)={trans_matrix};
    M_t=trans_matrix;
    tic;
    %[delta_s,fern(t,:),la(:,t),delta_la(:,:,t)]=LearnStageRegressor(Y,I,S_t{t},M_t{t},74,400,15,5,K);
    [delta_s,fern(t,:),la(:,t),delta_la(:,:,t)]=LearnStageRegressor(Y,I,S_t,M_t,N_aug,74,400,15,5,K);
    
    fprintf('internal time');
    toc
    %%%%%%%LearnStageRegressor
%     delta_s=zeros(N,148);
%     ShapeIndexedFeatures=GenerateShapeIndexedFeatures(I,S_t{t},M_t{t},74,400,5); 
%     cov_feat=cov(ShapeIndexedFeatures);
%     %out=zeros(10,1);
%     %R=zeros(32,148,'single');
%     cov_feat=cov(ShapeIndexedFeatures);
%     tic;
%     for k=1:500
%     [I_idx,DiffFeatures] = CorrelationBasedFeature( Y,ShapeIndexedFeatures,cov_feat,400,5 );
%     %norm(Y,'fro')
%     c=single(max(abs(DiffFeatures(:))));
%     theta=0.4*c.*(rand(1,F)-0.5);
%     partition=bsxfun(@gt,single(DiffFeatures),theta);
%     bin_num=sum(bsxfun(@times,partition,encoder),2)+1;
%     %size(partition)
%     omega_b=zeros(2^F,1);
%     y_b=zeros(2^F,148);
%     for i=1:7258
%         omega_b(bin_num(i))=omega_b(bin_num(i))+1;
%         y_b(bin_num(i),:)=y_b(bin_num(i),:)+Y(i,:);
%     end
%     omega_b=omega_b+1000;
%     regressor=bsxfun(@rdivide,y_b,omega_b);
%     for n=1:N
%         Y(n,:)=Y(n,:)-regressor(bin_num(n),:);
%         delta_s(n,:)=delta_s(n,:)+regressor(bin_num(n),:);
%     end
% end
    %%%%%%%LearnStageRegressor
    
    
    tic;
    if t<T
        %S_t(t+1)=S_t(t);
        for n=1:N
            for k=1:N_aug
                idx=(n-1)*N_aug+k;
                xxx=reshape(delta_s(idx,:),[74,2])/(trans_matrix{idx,1}*trans_matrix{idx,2});
                %S_t{t+1}{n,1}=S_t{t+1}{n,1}+reshape(xxx,[1,148]);
                S_t{n,k}=S_t{n,k}+reshape(xxx,[1,148]);
            end
        end
        %S_t(t+1)=S_t(t)+;
    end
    %size(S_t{t+1}{:,1}
    %out(t)=norm(reshape(S0,[7258,148])-S_t{t+1}{:,1},'fro');
    toc
end
