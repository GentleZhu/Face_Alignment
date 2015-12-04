function [ S_return ] = ESRTesting( I,meanshape,fern,init_scope,la,delta_la,N_int,init_set )
%ESRTesting Explicit Shape Regression Testing function
%   此处显示详细说明
N=size(I,1);
T=size(la,2);
S_t=cell(T,1);
M_t=cell(T,1);
%Y=zeros(N,148,'single');
trans_matrix=cell(N,3);
S_t(1)={Initialization(N,N_int,init_scope,init_set)};
%S_t(1)=init_set;

    
for t=1:T
    if t<T
        S_t(t+1)=S_t(t);
    end
    for k=1:N_int
%S_t(t)
        for n=1:N
            [~,~,trans] = procrustes(meanshape,reshape(S_t{t}{n,k},[74,2]));
            trans_matrix(n,:)=struct2cell(trans)';
        end
        M_t(t)={trans_matrix};
        delta_y=ApplyStageRegressor( I,S_t{t}(:,k),fern(t,:),M_t{t},la(:,t),delta_la(:,:,t),74,400,5,500);
        if t<T     
            for n=1:N
                xxx=reshape(delta_y(n,:),[74,2])/(trans_matrix{n,1}*trans_matrix{n,2});
                S_t{t+1}{n,k}=S_t{t+1}{n,k}+reshape(xxx,[1,148]);
            end
        end
    end
end

S_return=zeros(N,T,148,'single');
for n=1:N
    for t=1:T
        S_return(n,t,:)=(S_t{t}{n,1}+S_t{t}{n,2}+S_t{t}{n,3}+S_t{t}{n,4}+S_t{t}{n,5})/5;
    end
end

end

