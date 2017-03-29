function [net, info] = yolo_pd_train(varargin)
% yolo_pd_train fine-tunes a pre-trained CNN on imagenet dataset


run(fullfile(fileparts(mfilename('fullpath')), ...
  '..','externel', 'matconvnet','matlab', 'vl_setupnn.m')) ;
addpath('bbox_functions');
opts.dataDir = fullfile(fileparts(mfilename('fullpath')), '..\..','data');
opts.modelPath = fullfile(opts.dataDir, 'models', 'imagenet-vgg-f.mat');
opts.expDir  = fullfile(opts.dataDir, 'exp_pd_2') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.train = struct() ;
opts.train.gpus = [1];
opts.train.batchSize = 128 ;
opts.train.numSubBatches = 2 ;
opts.train.continue = true ;
opts.train.learningRate = 1e-3 * [ones(1,100), 0.5*ones(1,20), 0.1*ones(1,20), 0.01*ones(1,50)];
opts.train.numEpochs = 135 ;
opts.train.weightDecay = 0.0005 ;
opts.train.derOutputs = {'yololoss', 1} ;
opts.train.expDir = opts.expDir ;
opts.param.side = 7;   % for 7*7 grid cells
opts.param.locations = opts.param.side^2;% 49 grid cells

opts.param.coords = 4; % num of coords
opts.param.n = 2;      % each grid cell predicts n bbx 
opts.param.usegpu = opts.train.gpus;
opts.inputsize = [455 455];
opts.numFetchThreads = 2 ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%    Network initialization
% -------------------------------------------------------------------------
% net = yolo_pd_net_init('modelPath',opts.modelPath ,'param',opts.param);
net = yolo_pd_net_init_conc('modelPath',opts.modelPath,'param',opts.param);
% net = yolo_pd_net_init_tiny('modelPath',opts.modelPath,'param',opts.param);
% -------------------------------------------------------------------------
%   Database initialization
% -------------------------------------------------------------------------
if exist(opts.imdbPath,'file')
  fprintf('Loading imdb...');
  imdb = load(opts.imdbPath) ;
else
  if ~exist(opts.expDir,'dir')
    mkdir(opts.expDir);
  end
  imdb = yolo_pd_setup_data_cam('dataDir', opts.dataDir,'param',opts.param) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
fprintf('done\n');
% % % 求训练集的均值
% imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
% if exist(imageStatsPath,'file')
%   load(imageStatsPath, 'averageImage') ;
% else
%     averageImage = getImageStats(opts, imdb) ;
%     save(imageStatsPath, 'averageImage') ;
% end
% % % 用新的均值改变均值
% net.meta.normalization.averageImage = averageImage;

% --------------------------------------------------------------------
% Train
% --------------------------------------------------------------------
% use train + val + test split to train
% imdb.images.set(imdb.images.set == 2) = 1;

% minibatch options
bopts = opts.param;
bopts.numThreads = opts.numFetchThreads;
bopts.interpolation = net.meta.normalization.interpolation;
% bopts.averageImage = averageImage;
bopts.inputsize = opts.inputsize;
bopts.jitter = 0.2;
if numel(opts.train.gpus) >= 1 
  fprintf('%s: resetting GPU:', mfilename);
  tic;
  gpuDevice(opts.train.gpus);
  fprintf('%s: \n', toc);
end
[net,info] = cnn_train_dag(net, imdb, @(i,b) ...
                           getBatch(bopts,i,b), ...
                           opts.train) ;

% --------------------------------------------------------------------
% Deploy
% --------------------------------------------------------------------
modelPath = fullfile(opts.expDir, 'net-deployed.mat');
if ~exist(modelPath,'file')
  net = yolo_deploy(net);
  net_ = net.saveobj() ;
  save(modelPath, '-struct', 'net_') ;
  clear net_ ;
end

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

% -------------------------------------------------------------------------
function averageImage = getImageStats(opts, imdb)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
stride = train(1: 100: end);
avg = {};
for i = 1:length(stride)
    if stride(i)+99<length(train)
        temp = getBatchFn(opts,imdb, stride(i):stride(i)+99) ;
    else
        temp = getBatchFn(opts,imdb, stride(i):train(end)) ;
    end
    avg{end+1} = mean(temp, 4) ;
end
averageImage = mean(cat(4,avg{:}),4) ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
imo = vl_imreadjpeg(images,'numThreads',opts.numFetchThreads) ;
im  = zeros(opts.inputsize(1), opts.inputsize(2) , size(imo{1},3) , numel(batch),'single');
for b=1:numel(batch)
    imre = imresize(imo{b},[opts.inputsize(1) opts.inputsize(2)],'Method','bilinear');
    im(:,:,:,b) = single(imre);
end
fn = im;











