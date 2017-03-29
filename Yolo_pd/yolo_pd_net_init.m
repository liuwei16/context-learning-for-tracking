function net = yolo_pd_net_init(varargin)
% -------------------------------------------------------------------------
opts.modelPath = fullfile('data', 'models');
opts.param = struct();
opts = vl_argparse(opts, varargin) ;
net = load(opts.modelPath);
net = vl_simplenn_tidy(net);
% Skip layers after relu5
relu5p = find(cellfun(@(a) strcmp(a.name, 'relu5'), net.layers)==1);
net.layers{relu5p}.leak = 0.1;
net.layers = net.layers(1:relu5p);
% Add 2 conv layers 
% conv6
net.layers{end+1} = struct('name','conv6','type', 'conv', ...
                           'weights', {{0.001*randn(3,3,256,1024, 'single'), zeros(1, 1024, 'single')}}, ...
                           'stride', 1, ...
                           'pad', [1,1,1,1]) ;
net.layers{end+1} = struct('name','relu6','type', 'relu', 'leak', 0.1) ;
net.layers{end+1} = struct('name','pool6','type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0,1,0,1]) ;
% conv7                       
net.layers{end+1} = struct('name','conv7','type', 'conv', ...
                           'weights', {{0.001*randn(3,3,1024,1024, 'single'), zeros(1, 1024, 'single')}}, ...
                           'stride', 1, ...
                           'pad', [1,1,1,1]) ;
net.layers{end+1} = struct('name','relu7','type', 'relu', 'leak', 0.1) ;
net.layers{end+1} = struct('name','pool7','type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0,1,0,1]) ;
                       
% add batch normalization layers after conv6£¬7
% net = insertBnorm(net, 15, 'bnorm6') ;
% net = insertBnorm(net, 19, 'bnorm7') ;

% Add fc8 layer
net.layers{end+1} = struct('name','fc8','type', 'conv', ...
                           'weights', {{0.001*randn(7,7,1024,4096, 'single'), zeros(1, 4096, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('name','relu8','type', 'relu', 'leak', 0.1) ;
net.layers{end+1} = struct('name','drop8','type', 'dropout', 'rate', 0.5);
% Fill in defaul values
net = vl_simplenn_tidy(net) ;
% Convert to DagNN.
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

num_pred = floor(opts.param.side^2*(opts.param.n*(opts.param.coords+1)+opts.param.classes));
% Add prediction layer
pdrop8 = (arrayfun(@(a) strcmp(a.name, 'drop8'), net.layers)==1);
net.addLayer('pred', dagnn.Conv('size',[1 1 4096 num_pred],'hasBias', true), ...
net.layers(pdrop8).outputs{1}, 'prediction',{'predf','predb'}) ;
net.params(end-1).value = 0.001 * randn(1,1,4096,num_pred,'single');
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





