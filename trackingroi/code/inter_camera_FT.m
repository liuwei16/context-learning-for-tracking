function net = inter_camera_FT(net,idf,traj,opts)
% inter_camera_FT learns context infor when single target disappears

v_path = opts.v_path;
param = opts.param;
numCam = opts.numCam;
step = opts.step;
expDir = fullfile(opts.dataDir,'exp_pd_2');

% -------------------------------------------------------------------------
%   for sub componet experiment
% -------------------------------------------------------------------------
%    prepare the network
% -------------------------------------------------------------------------
% % Add yolo loss layer
% layer = net.layers(end);
% net.removeLayer(layer.name);
% net.addLayer('loss', dagnn.yoloLoss2('param',opts.param), ...
% {'prediction','truth'}, 'yololoss',{}) ;
% net.rebuild();
% for i = 7:10
%   net.params(i).weightDecay = 0;
%   net.params(i).learningRate = 0;
% end
% % -------------------------------------------------------------------------
% for i = 11:18
%   net.params(i).weightDecay = 0;
%   net.params(i).learningRate = 0;
% end
% -------------------------------------------------------------------------
%   Database initialization
% -------------------------------------------------------------------------
imdb = setup_data_forFT(idf,v_path,numCam,step,traj,param) ;
% opts.train.batchSize = numel(imdb.images.name);
opts.train.batchSize = 128 ;
opts.train.numSubBatches = 2 ;
opts.train.loss_thre = 0.1;


% minibatch options
bopts = opts.param;
bopts.numThreads = opts.numFetchThreads;
bopts.interpolation = net.meta.normalization.interpolation;
bopts.inputsize = opts.inputsize;
tic;
[net,~] = cnn_train_dag(net, imdb, @(i,b)getBatch(bopts,i,b), opts.train) ;
time = toc; 
a = time;
% % --------------------------------------------------------------------
% % Deploy
% % --------------------------------------------------------------------
% modelPath = fullfile(expDir, 'fnet_icl.mat');
% if ~exist(modelPath,'file')
%   net = yolo_deploy(net);
%   net_ = net.saveobj() ;
%   save(modelPath, '-struct', 'net_') ;
%   clear net_ ;
% end
% function imdb = setup_data_forFT(idf,v_path,numCam,step,traj,param)
% imdb.imageDir = fullfile(v_path,'images') ;
% -------------------------------------------------------------------------
% read Images
% -------------------------------------------------------------------------
% traj = traj(end-100,end,:);
traj = traj(traj(:,8)>0.5,:);
if size(traj,1)>300
   traj = traj(end-300:end,:) ;
end
numA = size(traj,1);
imname = cell(numCam*step+numA,1);
imsize = [240 320];
set = ones(numCam*step+numA,1);
gtbox = cell(numCam*step+numA,1);
gtlabel = cell(numCam*step+numA,1);
truth = cell(numCam*step+numA,1);
for i = 1:numA
    imname{i} = sprintf('%s%05d.jpg',num2str(traj(i,1)),traj(i,2)+1);
    gtbox{i} = [traj(i,4:5) traj(i,6:7)-traj(i,4:5)];
    gtlabel{i} = ones(1,1);
    truth{i} = truth_generate_cam(gtbox{i},gtlabel{i},imsize,param);
end
for i = 1:numCam
    for j = idf+1:idf+step
        imname{numA+(i-1)*step + j-idf} = sprintf('%s%05d.jpg',num2str(i),j);
        gtbox{numA+(i-1)*step + j-idf} = [0 0 0 0];
        gtlabel{numA+(i-1)*step + j-idf} = 0;
        truth{numA+(i-1)*step + j-idf} = zeros(1,1,5*param.side^2);       
    end    
end

imdb.images.name = imname;
imdb.images.size = imsize;
imdb.images.set = set;
imdb.boxes.gtbox = gtbox;
imdb.boxes.gtlabel = gtlabel;
imdb.boxes.truth = truth;


% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% da------------------------------------------------------------------
% images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
% [im, truth] = yolo_pd_train_get_batch(images, imdb, batch, opts);
% --------------------------------------------------------------------
% nda------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
ims = vl_imreadjpeg(images,'numThreads',opts.numThreads) ;
im  = zeros(opts.inputsize(1), opts.inputsize(2), size(ims{1},3) , numel(batch),'single');
s = size(imdb.boxes.truth{1},3);
truth = zeros(1,1,s,numel(batch));
for b=1:numel(batch)
    ims{b} = imresize(ims{b},[opts.inputsize(1) opts.inputsize(2)],'Method',opts.interpolation);
%     if ~isempty(opts.averageImage)
%         ims{b} = single(bsxfun(@minus,ims{b},opts.averageImage));       
%     end
    im(:,:,:,b) = ims{b};   
    truth(:,:,:,b) = imdb.boxes.truth{batch(b)};   
end
% --------------------------------------------------------------------
if numel(opts.usegpu) > 0
  im = gpuArray(im) ;
  truth = gpuArray(truth) ;
end
inputs = {'input', im, 'truth', truth} ;

% % --------------------------------------------------------------------
% function net = yolo_deploy(net)
% % --------------------------------------------------------------------
% for l = numel(net.layers):-1:1
%   if isa(net.layers(l).block, 'dagnn.yoloLoss') || ...
%       isa(net.layers(l).block, 'dagnn.DropOut')
%     layer = net.layers(l);
%     net.removeLayer(layer.name);
%     net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
%   end
% end
% net.rebuild();
% net.mode = 'test' ;








