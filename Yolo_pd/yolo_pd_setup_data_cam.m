function imdb = yolo_pd_setup_data_cam(varargin)
% Setup collected data for training
%
addpath('code');
opts.dataDir = 'data' ;
opts.param = struct();
opts.addFlipped = false ;
opts.useDifficult = true ;
opts = vl_argparse(opts, varargin) ;

imdb.classes.name = 'person';
imdb.classes.description = 'person' ;
imdb.imageDir = fullfile(opts.dataDir, 'CLTdataset', 'dataset2','train','images') ;
imdb.gtDir = fullfile(opts.dataDir, 'CLTdataset', 'dataset2','train','groundtruth') ;
% -------------------------------------------------------------------------
% read Images and gtboxes
% -------------------------------------------------------------------------
d_im = dir(fullfile(imdb.imageDir, '*.jpg'))';
d_gt = dir(fullfile(imdb.gtDir, '*.txt'))';
gtname = cell(numel(d_gt),1);
for i = 1:numel(d_gt)
    gtname{i} = d_gt(i).name;
end

imagesize = size(imread(fullfile(imdb.imageDir,d_im(1).name)));
imname = cell(numel(d_im),1);
imsize = zeros(numel(d_im),2);
set = ones(numel(d_im),1);
gtbox = cell(numel(d_im),1);
gtlabel = cell(numel(d_im),1);
yolobox = cell(numel(d_im),1);
truth = cell(numel(d_im),1);
for i = 1:numel(d_im)
    imname{i} = d_im(i).name;
    imsize(i,:) = imagesize([1 2]);
    if ismember([imname{i}(1:end-4) '.txt'], gtname)
       temp = load([imdb.gtDir '\' imname{i}(1:end-4) '.txt']);
       gtbox{i} = temp(:,2:end);
       gtlabel{i} = ones(size(gtbox{i},1),1);
       yolobox{i} = yolobox_generate_cam(gtbox{i},imsize(i,:));
       truth{i} = truth_generate_cam(gtbox{i},gtlabel{i},imsize(i,:),opts.param);
    else
       gtbox{i} = [0 0 0 0];
       gtlabel{i} = 0;
       yolobox{i} = [0 0 0 0];
       truth{i} = zeros(1,1,(5)*opts.param.side^2);
    end
end
imdb.images.name = imname;
imdb.images.size = imsize;
imdb.images.set = set;
imdb.boxes.gtbox = gtbox;
imdb.boxes.gtlabel = gtlabel;
imdb.boxes.yolobox = yolobox;
imdb.boxes.truth = truth;

% -------------------------------------------------------------------------
%  Flipped
% -------------------------------------------------------------------------
% imdb.boxes.flip = zeros(size(imdb.images.name));
% 
% imdb.images.name = vertcat(imdb.images.name, imdb.images.name) ;
% imdb.images.set  = vertcat(imdb.images.set, imdb.images.set) ;
% imdb.images.size  = vertcat(imdb.images.size, imdb.images.size) ;
% 
% imdb.boxes.gtbox = vertcat(imdb.boxes.gtbox , imdb.boxes.gtbox) ;
% imdb.boxes.gtlabel = vertcat(imdb.boxes.gtlabel, imdb.boxes.gtlabel) ;
% imdb.boxes.yolobox = vertcat(imdb.boxes.yolobox , imdb.boxes.yolobox) ;
% imdb.boxes.truth = vertcat(imdb.boxes.truth , imdb.boxes.truth) ;
% imdb.boxes.flip = vertcat(imdb.boxes.flip, ones(size(imdb.images.name))) ;
% for i=1:numel(imdb.boxes.gtbox)
%     if imdb.boxes.flip(i)
%       imf = imfinfo([imdb.imageDir filesep imdb.images.name{i}]);
%       gtbox = imdb.boxes.gtbox{i} ;
%       if ~isempty(gtbox)
%           assert(all(gtbox(:,1)<=imf.Width));
%           assert(all(gtbox(:,3)<=imf.Width));
%           gtbox(:,1) = imf.Width - gtbox(:,3);
%           gtbox(:,3) = imf.Width - imdb.boxes.gtbox{i}(:,1);
%           imdb.boxes.gtbox{i} = gtbox;
%           imdb.boxes.yolobox{i} = yolobox_generate(imdb.boxes.gtbox{i} , imdb.images.size(i,:));
%           imdb.boxes.truth{i} =
%           truth_generate_cam(imdb.boxes.gtbox{i},imdb.boxes.gtlabel{i},imdb.images.size(i,:),opts.param);%error
%       end      
%     end
% end
% -------------------------------------------------------------------------
%  Postprocessing
% -------------------------------------------------------------------------
[~,si] = sort(imdb.images.name) ;
imdb.images.name = imdb.images.name(si) ;
imdb.images.set = imdb.images.set(si) ;
imdb.images.size = imdb.images.size(si,:) ;
imdb.boxes.gtbox = imdb.boxes.gtbox(si)' ;
imdb.boxes.gtlabel = imdb.boxes.gtlabel(si) ;
imdb.boxes.yolobox = imdb.boxes.yolobox(si) ;
imdb.boxes.truth = imdb.boxes.truth(si) ;
% imdb.boxes.flip = imdb.boxes.flip(si) ;


