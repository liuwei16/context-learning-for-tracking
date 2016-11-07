%% initialise the tracker

% load videos
v{1}=VideoReader('../CLTdataset/dataset3-4cameras/Cam1.avi');
v{2}=VideoReader('../CLTdataset/dataset3-4cameras/Cam2.avi');
v{3}=VideoReader('../CLTdataset/dataset3-4cameras/Cam3.avi');
v{4}=VideoReader('../CLTdataset/dataset3-4cameras/Cam4.avi');
num = 0;
for i=1:4
    if  v{i}.NumberOfFrames>num
         num = v{i}.NumberOfFrames;
    end 
end

% load annotations
annotation_path='../CLTdataset/annotation_files/annotation/Dataset3/at least 4/';
annotation_files = dir([annotation_path '*.mat']);