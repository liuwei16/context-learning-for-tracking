function f = feat_extractor(frame,box,net)
%% extract the CNN features (using VGG model)

% Resize images and box to be compatible with the network.
imageSize = size(frame) ;
fullImageSize = net.meta.normalization.imageSize(1) ...
    / net.meta.normalization.cropSize ;
scale = max(fullImageSize ./ imageSize(1:2)) ;
imNorm = imresize(frame, scale, ...
              net.meta.normalization.interpolation, ...
              'antialiasing', false) ;
imNorm = bsxfun(@minus, imNorm, net.meta.normalization.averageImage) ;

box=single([box(:,[1,2]) box(:,[1,2])+box(:,[3,4])]')+1;
box = bsxfun(@times, box - 1, scale) + 1 ;
roi=[ones(1,size(box,2)) ; box];
% obtain the CNN otuput
net.conserveMemory = 0; 
net.eval({'data', imNorm, 'rois',roi}) ;
f = squeeze(gather(net.vars(net.getVarIndex('fc7x')).value)) ;        
