function [im, truth] = yolo_pd_train_get_batch(images, imdb, batch, opts)
%  Generates mini-batches for yolo train

imo = vl_imreadjpeg(images,'numThreads',opts.numThreads) ;
im  = zeros(opts.inputsize(1), opts.inputsize(2) , size(imo{1},3) , numel(batch),'single');
truth = zeros(1,1,(5+opts.classes)*locations,numel(batch));
% get training images by flip and jitter
for b=1:numel(batch)
  imSize = size(imo{b});
  ow = imSize(2);
  oh = imSize(1);
  dw = floor(ow*opts.jitter);
  dh = floor(oh*opts.jitter);
  pleft  = randi([-dw, dw]);
  pright = randi([-dw, dw]);
  ptop   = randi([-dh, dh]);
  pbot   = randi([-dh, dh]);
  swidth =  ow - pleft - pright;
  sheight = oh - ptop - pbot;
  sx = swidth  / ow;
  sy = sheight / oh;
  dx = pleft / swidth;
  dy = ptop / sheight;
  rect = [max(0,pleft) max(0,ptop) min(swidth,ow) min(sheight,oh)];
  imcp = imcrop(imo{b},rect);
  imre = imresize(imcp,[opts.inputsize(1) opts.inputsize(2)],'Method',opts.interpolation);
  if imdb.boxes.flip(batch(b))
    temp = imre;  
    imre = temp(:,end:-1:1,:);
  end
  im(:,:,:,b) = single(imre);  
  truth(:,:,:,b) = fill_truth_region(imdb.boxes.yolobox{batch(b)},imdb.boxes.gtlabel{batch(b)},dx,dy,1/sx,1/sy,opts);

end


function truth = fill_truth_region(yolobox,labels,dx,dy,sx,sy,opts)

temp = zeros((5+opts.classes)*opts.side^2,1);
% read and correct yolobox
for i = 1 : size(yolobox,1)
    left = yolobox(i,1)-yolobox(i,3)/2;
    right = yolobox(i,1)+yolobox(i,3)/2;
    top = yolobox(i,2)-yolobox(i,4)/2;
    bottom = yolobox(i,2)+yolobox(i,4)/2;
    
    left   = constrain(0 , 1 , left * sx -dx);
    right  = constrain(0 , 1 , right * sx -dx);
    top    = constrain(0 , 1 , top  * sy -dy);
    bottom = constrain(0 , 1 , bottom* sy - dy);
    
    x = (left+right)/2;
    y = (top+bottom)/2;
    w = constrain(0 , 1 , right - left);
    h = constrain(0 , 1 , bottom - top); 
    if w < 0.01 || h < 0.01, continue; end
    col = ceil(x*opts.side);
    row = ceil(y*opts.side);
    x = x*opts.side - (col-1);
    y = y*opts.side - (row-1);
    index = (col-1+(row-1)*opts.side)*(5+opts.classes);
    temp(index+1) = 1;
    temp(index+1+labels(i)) = 1;
    temp(index+opts.classes+2) = x;
    temp(index+opts.classes+3) = y;
    temp(index+opts.classes+4) = w;
    temp(index+opts.classes+5) = h;    
end
truth = reshape(temp,1,1,size(temp,1));


function b = constrain(min , max , a)
b = a;
if a < min, b = min; end
if a > max, b = max; end
