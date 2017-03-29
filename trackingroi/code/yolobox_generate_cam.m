function  yolobox = yolobox_generate_cam(gtbox,imgsize)

[row,clom]=size(gtbox);
yolobox = zeros(row,clom);
yolobox(:,1) = (gtbox(:,1) + gtbox(:,3)/2)/imgsize(2);
yolobox(:,2) = (gtbox(:,2) + gtbox(:,4)/2)/imgsize(1);
yolobox(:,3) = gtbox(:,3)/imgsize(2);
yolobox(:,4) = gtbox(:,4)/imgsize(1);






