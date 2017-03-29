function net = yolo_pd_net_init_conc(varargin)
% -------------------------------------------------------------------------
f = 0.001;
filter = 42;
filters = filter*3;
opts.modelPath = fullfile('data', 'models');
opts.param = struct();
opts = vl_argparse(opts, varargin) ;
net = load(opts.modelPath);
net = vl_simplenn_tidy(net);
% Skip layers after relu5
relu5p = find(cellfun(@(a) strcmp(a.name, 'relu5'), net.layers)==1);
net.layers = net.layers(1:relu5p);
% Fill in defaul values
net = vl_simplenn_tidy(net) ;
% Convert to DagNN.
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
% concatenate conv1 3 5 
net.addLayer('pool_c1', dagnn.Pooling('poolSize',[4 4],'stride',4), ...
'x3', 'pc1') ;
net.addLayer('conv_c1', dagnn.Conv('size',[3 3 64 filter],'hasBias', true,'pad', [1,1,1,1]), ...
'pc1', 'conc1',{'conv_c1f','conv_c1b'}) ;
net.addLayer('relu_c1', dagnn.ReLU('leak',0.1), ...
'conc1', 'rc1') ;
net.params(end-1).value = f * randn(3,3,64,filter,'single');
net.params(end).value = zeros(1,filter,'single');

net.addLayer('conv_c3', dagnn.Conv('size',[3 3 256 filter],'hasBias', true,'pad', [1,1,1,1]), ...
'x10', 'conc3',{'conv_c3f','conv_c3b'}) ;
net.addLayer('relu_c3', dagnn.ReLU('leak',0.1), ...
'conc3', 'rc3') ;
net.params(end-1).value = f * randn(3,3,256,filter,'single');
net.params(end).value = zeros(1,filter,'single');

net.addLayer('conv_c5', dagnn.Conv('size',[3 3 256 filter],'hasBias', true,'pad', [1,1,1,1]), ...
'x14', 'conc5',{'conv_c5f','conv_c5b'}) ;
net.addLayer('relu_c5', dagnn.ReLU('leak',0.1), ...
'conc5', 'rc5') ;
net.params(end-1).value = f * randn(3,3,256,filter,'single');
net.params(end).value = zeros(1,filter,'single');

% Add 2 conv layers 
% conv6
net.addLayer('conv6', dagnn.Conv('size',[3 3 filters 1024],'hasBias', true,'stride',1,'pad', [1,1,1,1]), ...
{'rc1','rc3','rc5'}, 'c6',{'conv6f','conv6b'}) ;
net.addLayer('relu6', dagnn.ReLU('leak',0.1), ...
'c6', 'r6') ;
net.addLayer('pool6', dagnn.Pooling('poolSize',[2 2],'stride',2), ...
'r6', 'p6') ;
net.params(end-1).value = f * randn(3,3,filters,1024,'single');
net.params(end).value = zeros(1,1024,'single');
% conv7                       
net.addLayer('conv7', dagnn.Conv('size',[3 3 1024 1024],'hasBias', true,'stride',1,'pad', [1,1,1,1]), ...
'p6', 'c7',{'conv7f','conv7b'}) ;
net.addLayer('relu7', dagnn.ReLU('leak',0.1), ...
'c7', 'r7') ;
net.addLayer('pool7', dagnn.Pooling('poolSize',[2 2],'stride',2), ...
'r7', 'p7') ;
net.params(end-1).value = f * randn(3,3,1024,1024,'single');
net.params(end).value = zeros(1,1024,'single');
% Add fc8 layer
net.addLayer('fc8', dagnn.Conv('size',[7 7 1024 1024],'hasBias', true), ...
'p7', 'f8',{'fc8f','fc8b'}) ;
net.addLayer('relu8', dagnn.ReLU('leak',0.1), ...
'f8', 'r8') ;
net.addLayer('drop8', dagnn.DropOut(), ...
'r8', 'd8') ;
net.params(end-1).value = f * randn(7,7,1024,1024,'single');
net.params(end).value = zeros(1,1024,'single');

num_pred = floor(opts.param.side^2*(opts.param.n*(opts.param.coords+1)));
% Add prediction layer
pdrop8 = (arrayfun(@(a) strcmp(a.name, 'drop8'), net.layers)==1);
net.addLayer('pred', dagnn.Conv('size',[1 1 1024 num_pred],'hasBias', true), ...
net.layers(pdrop8).outputs{1}, 'prediction',{'predf','predb'}) ;
net.params(end-1).value = f * randn(1,1,1024,num_pred,'single');
net.params(end).value = zeros(1,num_pred,'single');

% Add yolo loss layer
net.addLayer('loss', dagnn.yoloLoss('param',opts.param), ...
{'prediction','truth'}, 'yololoss',{}) ;
net.rebuild();

% fix the weights before conv5, just fine-tune the added layers
for i=1:6
  net.params(i).weightDecay = 0;
  net.params(i).learningRate = 0;
end
% % No decay for bias and set learning rate to 2
% for v = [17,22,27,32,34]
%   net.params(v).weightDecay = 0;
%   net.params(v).learningRate = 2;
% end

net.meta.inputSize = [455 455 3] ;
net.meta.normalization.interpolation = 'bilinear';
net.meta.classes.name = {'aeroplane', 'bicycle', 'bird', ...
    'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', ...
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', ...
    'sofa', 'train', 'tvmonitor', 'background' };


% --------------------------------------------------------------------
function net = insertBnorm(net, l, name)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('name',name,'type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;





