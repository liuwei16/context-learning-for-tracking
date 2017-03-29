function yolo_pd_demo(varargin)

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..','externel', 'matconvnet','matlab', 'vl_setupnn.m')) ;
addpath('bbox_functions');
opts.dataDir = fullfile(fileparts(mfilename('fullpath')), '..\..','data');
opts.modelPath = fullfile(opts.dataDir, 'exp_pd_2', 'net-deployed.mat');
% opts.modelPath = fullfile(opts.dataDir, 'exp_pd_2', 'fnet_0.mat');
opts.classes = {'person'} ;
opts.gpu = [] ;
opts.confThreshold = 0.1 ;
opts.nmsThreshold = 0.3 ;
opts.param.side = 7;   % for 7*7 grid cells
opts.param.locations = opts.param.side^2;% 49 grid cells
opts.param.coords = 4; % num of coords
opts.param.n = 2;      % each grid cell predicts n bbx 
opts.inputsize = [455 455];
opts = vl_argparse(opts, varargin) ;

% Load the network and put it in test mode.
net = load(opts.modelPath) ;
net = dagnn.DagNN.loadobj(net);
net.mode = 'test' ;
net.meta.classes.name = {'person'};
% Mark  predictions as `precious` so they are not optimized away during evaluation.
net.vars(net.getVarIndex('prediction')).precious = 1 ;
% Load a test image and candidate bounding boxes.
im = single(imread('100110.jpg')) ;
% im = single(imread('0.jpg')) ;
imageSize = size(im) ;
imo = im; % keep original image 
% Resize images and boxes to a size compatible with the network.
im = imresize(im,[opts.inputsize(1) opts.inputsize(2)],'Method',net.meta.normalization.interpolation);
% Remove the average color from the input image.
% imNorm = bsxfun(@minus, im, net.meta.normalization.averageImage) ;

% Evaluate network either on CPU or GPU.
if numel(opts.gpu) > 0
  gpuDevice(opts.gpu) ;
  im = gpuArray(im) ;
  net.move('gpu') ;
end
net.conserveMemory = false ;
net.eval({'input', im});
% Extract box coordinates, confidence and class probabilities 
% x25 = squeeze(gather(net.vars(net.getVarIndex('x25')).value)) ;
% x26 = squeeze(gather(net.vars(net.getVarIndex('x26')).value)) ;
prediction = squeeze(gather(net.vars(net.getVarIndex('prediction')).value)) ;
[boxes, probs] = convert_predictions(prediction, opts.param);
cboxes = bbox_std(boxes , imageSize);
for i = 1:numel(opts.classes)
  c = strcmp(opts.classes{i}, net.meta.classes.name) ;
  cprobs = probs(c,:) ;
  cls_dets = [cboxes ; cprobs]' ;
  keep = bbox_nms(cls_dets, opts.nmsThreshold) ;% for Non-maximum suppression
  cls_dets = cls_dets(keep, :) ;
  sel_boxes = cls_dets(:,end) >= opts.confThreshold ;
  imo = bbox_draw(imo/255,cls_dets(sel_boxes,:));
  title(sprintf('Detections for class ''%s''', opts.classes{i})) ;
  
end

