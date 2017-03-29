function imdb = reid_setup_data_roi(varargin)
% Setup collected data for training
%
opts.dataDir = 'data' ;
opts = vl_argparse(opts, varargin) ;

imdb.imageDir = fullfile(opts.dataDir, 'CLTdataset', 'dataset2','personroipair_c6') ;
% -------------------------------------------------------------------------
% read Images
% -------------------------------------------------------------------------
d_im = dir(fullfile(imdb.imageDir, '*.mat'))';

imname = cell(numel(d_im),1);
set = ones(numel(d_im),1);
label = ones(numel(d_im),1);
for i = 1:numel(d_im)
    imname{i} = d_im(i).name;
    if imname{i}(1)=='n'
       label(i)=2; 
    end
end
imdb.images.name = imname;
imdb.images.set = set;
imdb.images.label = label;



