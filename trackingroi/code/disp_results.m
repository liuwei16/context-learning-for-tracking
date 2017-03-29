%% disp the results within images
if ~isempty(res)
    index_cam = res(1);
else
    index_cam = 0;
end
figure(2);
for i=1:opts.numCam
    subplot(2,2,i);
    im =  imread([opts.v_path, sprintf('/images/%s%05d.jpg',num2str(i),idf)]);
    if i==index_cam
       im = bbox_draw(im,res(4:end)); title(sprintf('Cam%d',i));      
    else
       imshow(im); title(sprintf('Cam%d',i));
    end
   if i==1
      text(2, 4, strcat('#',num2str(idf-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10); 
   end
end
pause(0.01)
 