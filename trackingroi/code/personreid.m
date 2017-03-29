function  recognition = personreid(proberp,frame,detection,roinet,rnet,opts)
if ~isempty(detection)
    rec=detection(:,4:7);
    scale = 455./[320 240];
    rec(:,[1 3])= rec(:,[1 3])*scale(1);
    rec(:,[2 4])= rec(:,[2 4])*scale(2);
    frame = imresize(single(frame),[455 455],'Method','bilinear');
    rois = single([ones(1,size(rec,1)) ; rec']) ;
    query = zeros(6,6,256,size(detection,1));
    con = zeros(6,6,256,size(detection,1));
    count = 1;
    for i = 1:size(detection,1)
        roinet.eval({'input', frame(:,:,:,detection(i,1)), 'rois', rois(:,i)});
        query(:,:,:,count)=squeeze(gather(roinet.vars(roinet.getVarIndex('xRP')).value)) ;
        con(:,:,:,count)=abs(query(:,:,:,count)-proberp);
        count = count+1;
    end
    % Evaluate network either on CPU or GPU.
%     if numel(opts.train.gpus) > 0
%       con = gpuArray(con) ;
%       rnet.move('gpu') ;
%     end
    con=single(con);
    rnet.eval({'input', con});
    score = squeeze(gather(rnet.vars(rnet.getVarIndex('score')).value)) ;
    [bestscore, best] = max(score(1,:));
    if bestscore > 0.9
        recognition = detection(best,:);
    else
        recognition = [];
    end
else
    recognition = [];
end
