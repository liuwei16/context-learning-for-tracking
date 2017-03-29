function [net, info] = reid_train_roi(varargin)
% reid_train train a CNN model from scrach


run(fullfile(fileparts(mfilename('fullpath')), ...
  '..','externel', 'matconvnet','matlab', 'vl_setupnn.m')) ;
opts.dataDir = fullfile(fileparts(mfilename('fullpath')), '..\..','data');
opts.expDir  = fullfile(opts.dataDir, 'exp_reidroi') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.networkType = 'dagnn' ;

opts.train = struct() ;
opts.train.gpus = [];
opts.train.batchSize = 512 ;
opts.train.numSubBatches = 2 ;
opts.train.continue = true ;
opts.train.learningRate = 1e-3 * [ones(1,30), 0.1*ones(1,10)];
opts.train.numEpochs = 40 ;
opts.train.weightDecay = 0.0005 ;
opts.train.expDir = opts.expDir ;

% opts.inputsize = [64 64];
opts.numFetchThreads = 2 ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;
% -------------------------------------------------------------------------
%    Network initialization
% -------------------------------------------------------------------------
net = reid_net_init_roi('networkType', opts.networkType);
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
  imdb = reid_setup_data_roi('dataDir', opts.dataDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
fprintf('done\n');

% minibatch options
bopts.numThreads = opts.numFetchThreads;
bopts.usegpu = opts.train.gpus;
bopts.networkType=opts.networkType;
if numel(opts.train.gpus) >= 1 
  fprintf('%s: resetting GPU:', mfilename);
  tic;
  gpuDevice(opts.train.gpus);
  fprintf('%s: \n', toc);
end
switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

[net,info] = trainfn(net, imdb, ...
                           getBatch(bopts), ...
                           opts.train) ;

% --------------------------------------------------------------------
% Deploy
% --------------------------------------------------------------------
modelPath = fullfile(opts.expDir, 'net-deployed.mat');
if ~exist(modelPath,'file')
  net =netdeploy(net);
  net_ = net.saveobj() ;
  save(modelPath, '-struct', 'net_') ;
  clear net_ ;
end

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(opts,x,y) ;
  case 'dagnn'
    fn = @(x,y) getDagNNBatch(opts,x,y) ;
end
% --------------------------------------------------------------------
function [im, labels] = getSimpleNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im  = zeros(6, 6, 1024 , numel(batch),'single');
labels = imdb.images.label(batch);
for b=1:numel(batch)
    load(images{b})
    im(:,:,:,b) =con;
end
% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im  = zeros(6, 6, 1024 , numel(batch),'single');
label = imdb.images.label(batch);
for b=1:numel(batch)
    load(images{b})
    im(:,:,:,b) =con;
end
% --------------------------------------------------------------------
if numel(opts.usegpu) > 0
  im = gpuArray(im) ;
end
inputs = {'input', im, 'label', label} ;

% --------------------------------------------------------------------
function net = netdeploy(net)
% --------------------------------------------------------------------
for l = numel(net.layers):-1:1
  if isa(net.layers(l).block, 'dagnn.Loss') || ...
      isa(net.layers(l).block, 'dagnn.DropOut')
    layer = net.layers(l);
    net.removeLayer(layer.name);
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
  end
end
net.rebuild();
net.mode = 'test' ;





