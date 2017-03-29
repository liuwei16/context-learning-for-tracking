function reid_cls_roi(varargin)

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..','externel', 'matconvnet','matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile(fileparts(mfilename('fullpath')), '..\..','data');
opts.modelPath = fullfile(opts.dataDir, 'exp_reidroi','net-deployed-c6.mat');
opts.gpu = [] ;
opts = vl_argparse(opts, varargin) ;

% Load the network and put it in test mode.
net = load(opts.modelPath) ;
net = dagnn.DagNN.loadobj(net);

net.addLayer('sm', dagnn.SoftMax(), ...
'prediction', 'score',{}) ;
net.mode = 'test' ;
net.vars(net.getVarIndex('score')).precious = 1 ;
net.conserveMemory = false ;
imageDir = fullfile('../..','data', 'CLTdataset', 'dataset1','personroipair_c6') ;
d_im = dir(fullfile(imageDir, '*.mat'))';
imname = cell(numel(d_im),1);
clslabel = ones(numel(d_im),1);
scores = ones(numel(d_im),1);
label = ones(numel(d_im),1);
for i = 1:numel(d_im)
    load([imageDir '/' d_im(i).name]) ;
    im = con ;
%     im = gpuArray(im);
    net.eval({'input', im});
    score = squeeze(gather(net.vars(net.getVarIndex('score')).value)) ;
    [bestscore, best] = max(score) ;
    scores(i) = bestscore; 
    clslabel(i)=best;
    imname{i} = d_im(i).name;
    if d_im(i).name(1)=='n', label(i)=2; end
end
accurcy = length(find(clslabel==label))/length(label);
res = cat(2,label,clslabel,scores);
disp(['accurcy = ',num2str(accurcy*100),'%']);





  


