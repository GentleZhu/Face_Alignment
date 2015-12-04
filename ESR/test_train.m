 %imgs = cell(10,1)

 for n=1:84
     I=reshape(test2_feat(n,:,:),[250,250]);
    % I=reshape(add_feat(n,:,:),[250,250]);
for i=1:2
    %subplot(2,5,i);
    
    %markerInserter1 = vision.MarkerInserter('Shape','Plus','BorderColor','black');
     %markerInserter = vision.MarkerInserter('Shape','Plus','BorderColor','white');
    %Pts = reshape(int16(t{i,1}{n,1}),[74,2]);
    Pts = reshape(int16(t(n,i*5,:)),[74,2]);
    %Pts=uint8(T);
    %J = step(markerInserter, I, Pts);
    %imgs(i)=J;
    imshow(I);
    hold on;
    plot(Pts(:,1),Pts(:,2), 'g.');
    input('');
    %subimage(J);
    %h = imshow(J, 'InitialMag',1000, 'Border','loose');
    %imshow(J);
end
% markerInserter = vision.MarkerInserter('Shape','Plus','BorderColor','white');
%Pts = reshape(label(n,:,:),[74,2]);
%J = step(markerInserter, I, Pts);
%imshow(J);
%input('');
end
 