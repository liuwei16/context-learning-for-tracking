function run_CLT()
%% Context learning based tracker across camera networks
%
%  Wei Liu (liuwei16@nudt.edu.cn); Jingjing Xiao (shine636363@sina.com)  Nov. 2016
% 

warning off
addpath(genpath('./'))
init_tracker
%load fast VGG
run(fullfile(fileparts(mfilename('fullpath')), ...
   'externel','matconvnet', 'matlab', 'vl_setupnn.m')) ;
opts.modelPath = fullfile(fileparts(mfilename('fullpath')), ...
'..', 'models', 'fast-rcnn-vgg16-pascal07-dagnn.mat') ;
opts.gpu = [] ;
net = load(opts.modelPath) ;
net = dagnn.DagNN.loadobj(net);
net.mode = 'test' ;
%-------------------------

for idf = 1: length(annotation_files   
    % load detection
    load([annotation_path annotation_files(idf).name]);
    id_detc=id;
    
    feature=cell(4,num);
    % load image 
    for k = 1 : num         
        for i=1:4
            frame{i}= single(read(v{i},k));                     
            x{i}=find(id_detc{i}(:,2)==k-1);
            if ~isempty(x{i})
                box=id_detc{i}(x{i},([4,5,6,7]));
                % extract the CNN features (using VGG model)
                f=get_features(frame{i},box,net);                    
                feature{i,k} = f ;              
            end
        end
        
        %-------------------------------------------
        % show detection
        disp_results
    end
end