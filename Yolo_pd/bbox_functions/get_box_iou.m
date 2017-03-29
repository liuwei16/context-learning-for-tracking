function iou = get_box_iou(a , b)
       w = overlap(a.x, a.w, b.x, b.w);
       h = overlap(a.y, a.h, b.y, b.h);
       if w<0 || h < 0
          intersection =0 ;
       else
          intersection = w*h;
       end;      
       union = a.w*a.h + b.w*b.h - intersection;
       iou = intersection/union;           
    end