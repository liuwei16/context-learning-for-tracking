function net = first_frame_FT(net,active_c,start_f,idp,opts)
% first_frame_FT fine-tunes a pre-trained model for single target

v_path = opts.v_path;
param = opts.param;
numCam = opts.numCam;

expDir = fullfile(opts.dataDir,'exp_pd_2');
modelPath = fullfile(expDir, sprintf('fnet_%s.mat',num2str(idp)));
if exist(modelPath,'file')
    net=dagnn.DagNN.loadobj(load(modelPath));
else
    % -------------------------------------------------------------------------
    %    prepare the network
    % -------------------------------------------------------------------------
    % Add yolo loss layer
    layer = net.layers(end);
    net.removeLayer(layer.name);
    net.addLayer('loss', dagnn.yoloLoss2('param',opts.param), ...
    {'prediction','truth'}, 'yololoss',{}) ;
    net.rebuild();
    for i = 7:10
      net.params(i).weightDecay = 0;
      net.params(i).learningRate = 0;
    end

    % -------------------------------------------------------------------------
    %   Database initialization
    % -------------------------------------------------------------------------
    imdb = setup_data_forFT(active_c,start_f,idp,v_path,numCam,param) ;
    opts.train.batchSize = numel(imdb.images.name);
    opts.train.loss_thre = 0.01;
    % minibatch options
    bopts = opts.param;
    bopts.numThreads = opts.numFetchThreads;
    bopts.interpolation = net.meta.normalization.interpolation;
    bopts.inputsize = opts.inputsize;
    tic;
    [net,~] = cnn_train_dag(net, imdb, @(i,b)getBatch(bopts,i,b), opts.train) ;
    time = toc; 
    a=time;
    % --------------------------------------------------------------------
    % Deploy
    % --------------------------------------------------------------------
    % modelPath = fullfile(expDir, sprintf('fnet_%s.mat',num2str(idp)));
    if ~exist(modelPath,'file')
      net = yolo_deploy(net);
      net_ = net.saveobj() ;
      save(modelPath, '-struct', 'net_') ;
      clear net_ ;
    end
end

function imdb = setup_data_forFT(active_c,start_f,idp,v_path,numCam,param)
imdb.imageDir = fullfile(v_path,'images') ;
imdb.gtDir = fullfile(v_path, 'groundtruth') ;
% -------------------------------------------------------------------------
% read Images and gtboxes
% -------------------------------------------------------------------------
numDA = 11; % data augumentation : generate numDA boxes surrounding the gt
imname = cell(numCam+numDA,1);
imsize = [240 320];
set = ones(numCam+numDA,1);
gtbox = cell(numCam+numDA,1);
gtlabel = cell(numCam+numDA,1);
% yolobox = cell(numCam,1);
truth = cell(numCam+numDA,1);
for i = 1:numCam
    imname{i} = sprintf('%s%05d.jpg',num2str(i),start_f);
    if i==active_c
       temp = load([imdb.gtDir '\' imname{i}(1:end-4) '.txt']); 
       gtbox{i} = temp(temp(:,1)==idp,2:end);
       gtlabel{i} = ones(1,1);
       truth{i} = truth_generate_cam(gtbox{i},gtlabel{i},imsize,param);
    else
        gtbox{i} = [0 0 0 0];
        gtlabel{i} = 0;
        truth{i} = zeros(1,1,5*param.side^2);
    end    
end
boxes = addbox(gtbox{active_c},numDA,imsize);
for i = numCam+1:numCam+numDA
    imname{i} = sprintf('%s%05d.jpg',num2str(active_c),start_f);
    gtbox{i} = boxes(i-numCam,:);
    gtlabel{i} = ones(1,1);
    truth{i} = truth_generate_cam(gtbox{i},gtlabel{i},imsize,param);
end
imdb.images.name = imname;
imdb.images.size = imsize;
imdb.images.set = set;
imdb.boxes.gtbox = gtbox;
imdb.boxes.gtlabel = gtlabel;
% imdb.boxes.yolobox = yolobox;
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

% --------------------------------------------------------------------
function net = yolo_deploy(net)
% --------------------------------------------------------------------
for l = numel(net.layers):-1:1
  if isa(net.layers(l).block, 'dagnn.yoloLoss') || ...
      isa(net.layers(l).block, 'dagnn.DropOut')
    layer = net.layers(l);
    net.removeLayer(layer.name);
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
  end
end
net.rebuild();
net.mode = 'test' ;

% --------------------------------------------------------------------
function boxes = addbox(box,numDA,imsize)
% --------------------------------------------------------------------
dw = floor(0.05*box(3));
dh = floor(0.05*box(4));
for i = 1 : numDA
    pw = randi([-dw, dw]);
    ph = randi([-dh, dh]);
    s = 0.9+0.2*rand(1,2);
    boxes(i,:) = [box(1)+pw box(2)+ph box(3)*s(1) box(4)*s(2)];
end
imW = imsize(2);
imH = imsize(1);
boxes(:,1) = max(0,boxes(:,1));
boxes(:,2) = max(0,boxes(:,2));
boxes(:,3) = min(imW-boxes(:,1),boxes(:,3));
boxes(:,4) = min(imH-boxes(:,2),boxes(:,4));







