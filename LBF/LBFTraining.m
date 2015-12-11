N=1000;
Nfp=74;
I=t_feat(1:N,:,:);
S0=t_label(1:N,1:Nfp,:);
meanshape_x=meanshape(1:Nfp,:);
N_aug=1;

init_scope=face_init(1:N,:,:);
init_set=t_label(1:N,:,:);
P=100;
%T=5;
T=1;
S_t=cell(T,1);

S_t{1}=Initialization(N,N_aug,Nfp,init_scope,init_set);
trans_matrix=cell(N*N_aug,3);
Y=zeros(N*N_aug,Nfp,2,'single');
y_b=zeros(N*N_aug,Nfp,2,'single');
S0=single(S0);
%F=5;
K=500;
out=zeros(T+1,1);
radius=[20 15 10 5 1];
%encoder=[1 2 4 8 16];
%fern=cell(T,K);
for t=1:T
    tic;
    for n=1:N
        idx=n;
        %for k=1:N_aug
        %idx=(n-1)*N_aug+k;
        [~,Ytmp,trans] = procrustes(meanshape_x,reshape(S_t{t}{n,1},[Nfp,2]));
        %size(S_t)
        %[~,Ytmp,trans] = procrustes(meanshape,reshape(S_t{n,k},[74,2]));
        trans_matrix(idx,:)=struct2cell(trans)';
        Y(idx,:,:)=trans_matrix{idx,2}*reshape(S0(n,:,:),[Nfp,2])*trans_matrix{idx,1}+trans_matrix{idx,3}-Ytmp;%,[1,148]);
        %end
    end
    out(t)=norm(reshape(Y(:,1,:),[N*N_aug,2]),'fro');
    toc
    M_t=trans_matrix;
    ShapeIndexedFeatures=GenerateShapeIndexedFeatures(I,S_t{t},M_t,1,Nfp,P,radius(t));
    %delta_s=zeros(N,Nfp*2);
    tic;
    for l=1:1
        fprintf('landmark %d\n',l);
        yy=reshape(Y(:,l,:),[N 2]);
        %out(1,1)=norm(yy,'fro');
        xx=reshape(ShapeIndexedFeatures(l,:,:),[N,P*P]);
        nodes= train_rfs( xx,yy,10,5,500);
         y_b(:,l,:)= reshape(test_rfs( nodes ,xx,10,5 ),[N,1,2]);
         %out(2,1)=norm(yy-y_b,'fro');
    end
    toc
    out(t+1)=norm(yy-reshape(y_b(:,1,:),[N,2]),'fro');
    %Y=Y-y_b;
    fprintf('Update!\n');
    for n=1:N
        xxx=reshape(y_b(n,:,:),[Nfp,2])/(trans_matrix{n,1}*trans_matrix{n,2});
        S_t{t+1}{n,1}=S_t{t}{n,1}+reshape(xxx,[1,2*Nfp]);
%         Y(n,:)=Y(n,:)-regressor(bin_num(n),:);
%         delta_s(n,:)=delta_s(n,:)+regressor(bin_num(n),:);
    end
end