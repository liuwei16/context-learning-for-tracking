function  truth = truth_generate_cam(gtbox,gtlabel,imgsize,param)

side = param.side;% there are 7*7 grid cells
location = [0:1:side]/side;
output = 5; % 1+4

temp=zeros(output , side^2); % for truth

% convert the annotations to boxes in yolo style
[row,clom]=size(gtbox);
yolobox = zeros(row,clom);
yolobox(:,1) = (gtbox(:,1) + gtbox(:,3)/2)/imgsize(2);
yolobox(:,2) = (gtbox(:,2) + gtbox(:,4)/2)/imgsize(1);
yolobox(:,3) = gtbox(:,3)/imgsize(2);
yolobox(:,4) = gtbox(:,4)/imgsize(1);

 for k=1:row % row is the num of boxes
    grid_truth = zeros(output,1);   
    grid_truth(1) = gtlabel(k); % confidence 
    
    px = find(location<yolobox(k,1), 1, 'last' );% px: index of col in 7*7
    py = find(location<yolobox(k,2), 1, 'last' );% py: index of row in 7*7
    grid_truth(2) = yolobox(k,1)*side-px+1;% x: offset of practical grid cell
    grid_truth(3) = yolobox(k,2)*side-py+1;% y: offset of practical grid cell
    grid_truth(4) = yolobox(k,3);               % w: relative to image size
    grid_truth(5) = yolobox(k,4);               % h: relative to image size

    tag = (py-1)*side+px; % index of grid cell     
    temp(:,tag) = grid_truth; % weight
 end
truth = reshape( temp , 1 , 1 , output*side^2 );




