function detection= perform_detection(frame,net,opts,idf,idp,active_c,conf)

opts.confThreshold = conf;
im = single(frame) ;
imageSize = size(im) ;
% Resize images and boxes to a size compatible with the network.
im = imresize(im,[opts.inputsize(1) opts.inputsize(2)],'Method',net.meta.normalization.interpolation);
% Evaluate network either on CPU or GPU.
if numel(opts.train.gpus) > 0
  im = gpuArray(im) ;
  net.move('gpu') ;
end
net.conserveMemory = false ;
net.eval({'input', im});
% Extract box coordinates, confidence and class probabilities 
prediction = squeeze(gather(net.vars(net.getVarIndex('prediction')).value)) ;
detection = [];
if size(prediction,2)==1
    [boxes, probs] = convert_predictions(prediction, opts.param);
    cboxes = bbox_std(boxes , imageSize);
    cls_dets = [cboxes ; probs]' ;
    keep = bbox_nms(cls_dets, opts.nmsThreshold) ;% for Non-maximum suppression
    cls_dets = cls_dets(keep, :) ;
    sel_boxes = find(cls_dets(:,end) >= opts.confThreshold) ;
    if ~isempty(sel_boxes)
       bbx = cls_dets(sel_boxes,:);
       detection =[detection; [active_c*ones(size(bbx,1),1) idf*ones(size(bbx,1),1) idp*ones(size(bbx,1),1) bbx]];
    end
else
    for i = 1:size(prediction,2)
%         if i==active_c, continue; end
        [boxes, probs] = convert_predictions(prediction(:,i), opts.param);
        cboxes = bbox_std(boxes , imageSize);
        cls_dets = [cboxes ; probs]' ;
        keep = bbox_nms(cls_dets, opts.nmsThreshold) ;% for Non-maximum suppression
        cls_dets = cls_dets(keep, :) ;
        sel_boxes = find(cls_dets(:,end) >= opts.confThreshold) ;
        if ~isempty(sel_boxes)
           bbx = cls_dets(sel_boxes,:);
           detection =[detection; [i*ones(size(bbx,1),1) idf*ones(size(bbx,1),1) idp*ones(size(bbx,1),1) bbx]];
        end
   end
end



