%% initialise the tracker
opts.dataDir = fullfile(fileparts(mfilename('fullpath')), '..\..\..','data');
opts.modelPath = fullfile(opts.dataDir, 'exp_pd_2', 'net-epoch-135.mat');
opts.reidmodelPath = fullfile(opts.dataDir, 'exp_reidroi', 'net-deployed-0.1.mat');
% load video path and compute num of frames
opts.numCam = 3;
opts.v_path = fullfile(opts.dataDir, 'CLTdataset', 'dataset1') ;
% v_path(v_path=='\')='/';
opts.numF = length(dir(fullfile(opts.v_path,'images', '*.jpg'))')/opts.numCam;
opts.step = 20; % the frames for inter camera finetuing 

opts.train.gpus = [1];
opts.train.batchSize = 4 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.learningRate = 1e-4*[ones(1,5) ones(1,5)];
opts.train.numEpochs = 10 ;
opts.train.weightDecay = 0.0005 ;
opts.train.derOutputs = {'yololoss', 1} ;
opts.train.saveState = false;
opts.train.plotStatistics = false;
opts.param.side = 7;   % for 7*7 grid cells
opts.param.locations = opts.param.side^2;% 49 grid cells
opts.param.coords = 4; % num of coords
opts.param.n = 2;      % each grid cell predicts n bbx
opts.param.usegpu = opts.train.gpus;
opts.inputsize = [455 455];
opts.numFetchThreads = 2 ;
opts.confThreshold = 0.6 ;
opts.nmsThreshold = 0.2 ;
% Load the pretrained network.
pnet = load(opts.modelPath) ;
pnet = dagnn.DagNN.loadobj(pnet.net);
% init the gpudevice
if numel(opts.train.gpus) >= 1 
  fprintf('%s: resetting GPU:', mfilename);
  tic;
  gpuDevice(opts.train.gpus);
  fprintf('%s: \n', toc);
end
%load cameras' detection
cam{1}  = load('../../data/CLTdataset/annotation_files/annotation/Dataset1/Cam1.dat');
cam{2}  = load('../../data/CLTdataset/annotation_files/annotation/Dataset1/Cam2.dat');
cam{3}  = load('../../data/CLTdataset/annotation_files/annotation/Dataset1/Cam3.dat');
org_det  = [cam{1}; cam{2}; cam{3}]; % sort out sequential orders
[~, idd] = sort(org_det(:, 2)); 
det      = org_det(idd, :);
numP     = max(org_det(:, 3)); % number of pedestrian

% init model for roi feature extraction
roinet = load(opts.modelPath) ;
roinet = dagnn.DagNN.loadobj(roinet.net);
pRelu3 = find(arrayfun(@(a) strcmp(a.name, 'relu3'), roinet.layers)==1);
roinet.layers = roinet.layers(1:pRelu3);
roinet.params = roinet.params(1:6);
roinet.vars = roinet.vars(1:11);
roinet.addLayer('roipool', dagnn.ROIPooling('method','max','transform',1/16,...
    'subdivisions',[6,6],'flatten',0), ...
    {roinet.layers(pRelu3).outputs{1},'rois'}, 'xRP');
roinet.conserveMemory = false ;

% init model for reid
rnet = dagnn.DagNN.loadobj(load(opts.reidmodelPath)) ;
rnet.addLayer('sm', dagnn.SoftMax(), 'prediction', 'score', {}) ;
rnet.mode = 'test' ;
rnet.vars(rnet.getVarIndex('score')).precious = 1 ;
rnet.conserveMemory = false ;
