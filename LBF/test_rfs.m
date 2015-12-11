function [ y_b ] = test_rfs( node ,X,num_trees,max_depth )
%test_rfs Get feature vector
N=size(X,1);
%index=1;
y_b=zeros(N,2,'single');
y_temp=zeros(num_trees,2,'single');
for i=1:N
    for k=1:num_trees
        index=1;
    for d=1:max_depth-1
        num_feat=node{k,index}.split_point;
        if X(i,num_feat)>node{k,index}.threshold
            index=2*index+1;
        else
            index=2*index;
        end
    end
    y_temp(k,:)=node{k,index};
    end
    y_b(i,:)=mean(y_temp,1);
end

