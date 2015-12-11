function [ nodes ] = train_rfs( X,Y,num_trees,max_depth,num_feat )
%train_rfs 此处显示有关此函数的摘要
%   此处显示详细说明
[N,P]=size(X);
k=500;
%index=zeros(max_depth,1);
nodes=cell(num_trees,2^max_depth-1);

for i=1:num_trees
    depth=1;
    nodes{i,1}=split_node(X,Y,randperm(P,num_feat),randperm(N,k),2^(max_depth-depth));
    for j=2:2^max_depth-1
        if j>=2^depth
            depth=depth+1;
        end
        feat=randperm(P,num_feat);
        parent=floor(j/2);
        %i
        %j
        %parent
        if j==parent*2
            nodes{i,j}=split_node(X,Y,feat,nodes{i,parent}.left_child,2^(max_depth-depth));
        else
            nodes{i,j}=split_node(X,Y,feat,nodes{i,parent}.right_child,2^(max_depth-depth));
        end
    end
end

end

function [tree_node]=split_node(X,Y,selected_feat,selected_sample,min_num)
if (min_num>1)
    subX=X(selected_sample,selected_feat)+randn(length(selected_sample),length(selected_feat));
    sort_x=sort(subX,1);
    subY=single(Y(selected_sample,:));
    max_i=0;
    max_threshold=0;
    max_loss=0;
    %fprintf('here\n');
    ind = randi([min_num,length(selected_sample)-min_num],length(selected_feat),1);
    for i=1:length(selected_feat)
        %iidx=selected_feat(i);
        %temp=subX(:,selected_feat(i));
        %range=int16(0.3*(max(temp)-min(temp)));
        %threshold=randi([-range/2 range/2]);
        threshold=sort_x(ind(i),i);
        partition=bsxfun(@gt,subX(:,i),threshold);
        left=subY(partition==1,:);
        right=subY(partition==0,:);
        %var(subY)-var(left)-var(right)
        reduction=norm(var(subY)-var(left)-var(right));
        %reduction
        if reduction>max_loss
            max_loss=reduction;
            max_i=i;
            max_threshold=threshold;
        end
    end
temp=subX(:,max_i);
partition=bsxfun(@gt,temp,max_threshold);
Param.split_point=selected_feat(max_i);
fprintf('Split point %d',selected_feat(max_i)); 
Param.right_child=selected_sample(partition==1);
Param.left_child=selected_sample(partition==0);
Param.threshold=max_threshold;
%if Param.right_child~=Param.left_child
%    fprintf('Error\n');
%end
tree_node=Param;
else
tree_node=mean(single(Y(selected_sample,:)));
end
end
