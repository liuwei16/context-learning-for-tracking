function run_CLT()
%% Context learning based tracker across camera networks
%
%  Wei Liu (XXX); Jingjing Xiao (shine636363@sina.com)  Nov. 2016
% 

warninig off
addpath(genpath('./'))
init_tracker
%-------------------------
%   load fast VGG model
%-------------------------

for idf = 1: length(annotation_files)
    
    % load detection
    load([annotation_path annotation_files(idf).name]);
    id_detc=id;
    
    %-------------------------------------------
    % extract the CNN features (using VGG model) 
    % of detections
    %-------------------------------------------
    
    % load image 
    for k = 1 : num
        frame1 = read(v{1},k); frame2 = read(v{2},k); frame3 = read(v{3},k); frame4 = read(v{4},k);    
        
        % show detection
        disp_results
    end
end