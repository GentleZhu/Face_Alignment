%load 'LFW.mat'
all=[t_label;add_label];
[N,d,~]=size(all);
meanshape=double(reshape(all(randi([1 N]),:,:),[74 2]));
normal_shape=zeros(N,74,2);
for k=1:5
    %diff=0;
    for i=1:N
        X=double(reshape(all(i,:,:),[74,2]));
        [~,normal_shape(i,:,:)] = procrustes(meanshape,X);
        %norm(meanshape-reshape(normal_shape(i,:,:),[74 2]),'fro')
        %d
        %w(i,:)=struct2cell(tmp)';
        %diff=diff+d;
    end
    %break;
    meanshape_2=reshape(mean(normal_shape,1),[74 2]);
    %meanshape_2=meanshape_2-repmat(mean(meanshape_2,2),1,size(meanshape_2,2));
    %meanshape_2=meanshape_2/norm(meanshape_2,'fro');
    diff=norm(meanshape-meanshape_2,'fro')
    %diff
    %if diff<0.00001
    %    break;
    %end
    meanshape=meanshape_2;
end
%test_procrustes
%for alpha=1:P