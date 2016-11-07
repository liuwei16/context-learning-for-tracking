%% disp the results within images

subplot(2,2,1); 
imshow(frame1),title('Cam1');
text(2, 4, strcat('#',num2str(k-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10);
x=find(id_detc{1}(:,2)==k-1);
if ~isempty(x)
    rectangle('Position',id_detc{1}(x,([4,5,6,7])),'LineWidth',2,'EdgeColor','r');
end
subplot(2,2,2);
imshow(frame2),title('Cam2');
text(2, 4, strcat('#',num2str(k-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10);
x=find(id_detc{2}(:,2)==k-1);
if ~isempty(x)
    rectangle('Position',id_detc{2}(x,([4,5,6,7])),'LineWidth',2,'EdgeColor','r');
end
subplot(2,2,3);
imshow(frame3),title('Cam3');
text(2, 4, strcat('#',num2str(k-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10);
x=find(id_detc{3}(:,2)==k-1);
if ~isempty(x)
    rectangle('Position',id_detc{3}(x,([4,5,6,7])),'LineWidth',2,'EdgeColor','r');
end
subplot(2,2,4);
imshow(frame4),title('Cam4');
text(2, 4, strcat('#',num2str(k-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10);
x=find(id_detc{4}(:,2)==k-1);
if ~isempty(x)
    rectangle('Position',id_detc{4}(x,([4,5,6,7])),'LineWidth',2,'EdgeColor','r');
end
pause(0.01)
clf  