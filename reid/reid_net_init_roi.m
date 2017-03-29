function net = reid_net_init_roi(varargin)
% reid_net_init Initialize a CNN similar to MNIST

opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;
rng('default');
rng(0) ;
f=1/100 ;
net.layers = {} ;
% net.layers{end+1} = struct('name','conv1','type', 'conv', ...
%                            'weights', {{f*randn(3,3,256,256, 'single'), zeros(1, 256, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('name','relu1','type', 'relu','leaky',0.1);
% net.layers{end+1} = struct('name','conv2','type', 'conv', ...
%                            'weights', {{f*randn(3,3,256,256, 'single'), zeros(1, 256, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', [1 1 1 1]) ;
% net.layers{end+1} = struct('name','relua','type', 'relu','leaky',0.1);

net.layers{end+1} = struct('name','fc1','type', 'conv', ...
                           'weights', {{f*randn(6,6,1024,4096, 'single'), zeros(1, 4096, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('name','relu2','type', 'relu','leaky',0.1);
net.layers{end+1} = struct('name','dp1','type', 'dropout', 'rate', 0.5);
net.layers{end+1} = struct('name','fc2','type', 'conv', ...
                           'weights', {{f*randn(1,1,4096,4096, 'single'), zeros(1, 4096, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('name','relu3','type', 'relu','leaky',0.1);
net.layers{end+1} = struct('name','dp2','type', 'dropout', 'rate', 0.5);
net.layers{end+1} = struct('name','pred','type', 'conv', ...
                           'weights', {{f*randn(1,1,4096,2, 'single'), zeros(1,2,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('name','sm','type', 'softmaxloss') ;


% Fill in defaul values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
      {'prediction', 'label'}, 'error') ;
  otherwise
    assert(false) ;
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
