function net = yolo_pd_net_init_tiny(varargin)
% -------------------------------------------------------------------------
f = 0.001;
opts.modelPath = fullfile('data', 'models');
opts.param = struct();
opts = vl_argparse(opts, varargin) ;
net = load(opts.modelPath);
net = vl_simplenn_tidy(net);
% Skip layers after relu3
relu3p = find(cellfun(@(a) strcmp(a.name, 'relu3'), net.layers)==1);
net.layers = net.layers(1:relu3p);
% Add 2 conv layers 
% conv4
net.layers{end+1} = struct('name','conv4','type', 'conv', ...
                           'weights', {{f*randn(3,3,256,1024, 'single'), zeros(1, 1024, 'single')}}, ...
                           'stride', 1, ...
                           'pad', [1,1,1,1]) ;
net.layers{end+1} = struct('name','relu4','type', 'relu', 'leak', 0.1) ;
net.layers{end+1} = struct('name','pool4','type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
% conv5                       
net.layers{end+1} = struct('name','conv5','type', 'conv', ...
                           'weights', {{f*randn(3,3,1024,1024, 'single'), zeros(1, 1024, 'single')}}, ...
                           'stride', 1, ...
                           'pad', [1,1,1,1]) ;
net.layers{end+1} = struct('name','relu5','type', 'relu', 'leak', 0.1) ;
net.layers{end+1} = struct('name','pool5','type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
% Add fc6 layer
net.layers{end+1} = struct('name','fc6','type', 'conv', ...
                           'weights', {{f*randn(7,7,1024,1024, 'single'), zeros(1, 1024, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('name','relu6','type', 'relu', 'leak', 0.1) ;
net.layers{end+1} = struct('name','drop6','type', 'dropout', 'rate', 0.5);
% Fill in defaul values
net = vl_simplenn_tidy(net) ;
% Convert to DagNN.
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
% Add prediction layer                  
num_pred = floor(opts.param.side^2*(opts.param.n*(opts.param.coords+1)));
pdrop6 = (arrayfun(@(a) strcmp(a.name, 'drop6'), net.layers)==1);
net.addLayer('pred', dagnn.Conv('size',[1 1 1024 num_pred],'hasBias', true), ...
net.layers(pdrop6).outputs{1}, 'prediction',{'predf','predb'}) ;
net.params(end-1).value = 0.001 * randn(1,1,1024,num_pred,'single');
net.params(end).value = zeros(1,num_pred,'single');
% Add yolo loss layer
net.addLayer('loss', dagnn.yoloLoss('param',opts.param), ...
{'prediction','truth'}, 'yololoss',{}) ;
net.rebuild();
% fix the weights before conv3, just fine-tune the added layers
for i=1:6
  net.params(i).weightDecay = 0;
  net.params(i).learningRate = 0;
end



net.meta.inputSize = [455 455 3] ;
net.meta.normalization.interpolation = 'bilinear';
net.meta.classes.name = {'person'};








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





