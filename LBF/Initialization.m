function [ initial_data ] = Initialization( N,N_aug,Nfp,init_scope,InitSet )
%Initialization Randomly select exemplar shapes from Initset
%   此处显示详细说明;
init_shape=randi([1 N],N,N_aug);
initial_data=cell(N,N_aug);
InitSet=single(InitSet);
for c=1:N
    w=init_scope(c,3);
    h=init_scope(c,4);
    center=init_scope(c,1:2);
%     target=reshape(init_scope(c,:,:),[74,2]);
%     center=mean(target);
%     w=max(target)-min(target);
    for d=1:N_aug
        n=init_shape(c,d);
        %n=100;
        candidate=reshape(InitSet(n,1:Nfp,:),[Nfp 2]);
        candidate=bsxfun(@minus,candidate,mean(candidate));
        t=max(candidate)-min(candidate);
        candidate=bsxfun(@times,candidate,[w/t(1),h/t(2)]);

%         candidate=bsxfun(@times,candidate,w./t);

        candidate=bsxfun(@plus,candidate,center);
        initial_data(c,d)={reshape(candidate,[1,Nfp*2])};
        %initial_data(c,d)={reshape(InitSet(init_shape(c,d),:,:),[1,148])};
    %initial_data(c,d)={reshape(InitSet(n,:,:),[1,148])};
    end
end
end

