function run_CLT()
%% Context learning based tracker across camera networks
%
%  Wei Liu (XXX); Jingjing Xiao (shine636363@sina.com)  Nov. 2016
% 

%addpath(genpath('./'))
c{1}=VideoReader('E:/Multi camera tracking/CLTdataset/dataset3-4cameras/Cam1.avi');
c{2}=VideoReader('E:/Multi camera tracking/CLTdataset/dataset3-4cameras/Cam2.avi');
c{3}=VideoReader('E:/Multi camera tracking/CLTdataset/dataset3-4cameras/Cam3.avi');
c{4}=VideoReader('E:/Multi camera tracking/CLTdataset/dataset3-4cameras/Cam4.avi');
num = 0;
for i=1:4
    if  c{i}.NumberOfFrames>num
         num = c{i}.NumberOfFrames;
    end 
end
% load annotations
annotation_path='E:/Multi camera tracking/CLTdataset/annotation_files/annotation/Dataset3/at least 4/';
annotation_files = dir([annotation_path '*.mat']);
for idf = 1: length(annotation_files)
    % load detection
    load([annotation_path annotation_files(idf).name]);
    id_detc=id;
    
    % load image 
    for k = 1 : num
        frame1 = read(c{1},k);
        frame2 = read(c{2},k);
        frame3 = read(c{3},k);
        frame4 = read(c{4},k);       
        % show detection
        subplot(2,2,1); 
        imshow(frame1),title('Cam1');
        text(2, 4, strcat('#',num2str(k-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10);
        x=find(id_detc{1}(:,2)==k-1);
        if ~isempty(x)
           rectangle('Position',id_detc{1}(x,([4,5,6,7])),'LineWidth',2,'EdgeColor','r');
        end
        subplot(2,2,2); 
        imshow(frame2),title('Cam2');
        text(2, 4, strcat('#',num2str(k-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10);
        x=find(id_detc{2}(:,2)==k-1);
        if ~isempty(x)
           rectangle('Position',id_detc{2}(x,([4,5,6,7])),'LineWidth',2,'EdgeColor','r');
        end
        subplot(2,2,3); 
        imshow(frame3),title('Cam3');
        text(2, 4, strcat('#',num2str(k-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10);
         x=find(id_detc{3}(:,2)==k-1);
        if ~isempty(x)
           rectangle('Position',id_detc{3}(x,([4,5,6,7])),'LineWidth',2,'EdgeColor','r');
        end
        subplot(2,2,4); 
        imshow(frame4),title('Cam4');
        text(2, 4, strcat('#',num2str(k-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10);
         x=find(id_detc{4}(:,2)==k-1);
        if ~isempty(x)
           rectangle('Position',id_detc{4}(x,([4,5,6,7])),'LineWidth',2,'EdgeColor','r');
        end
        pause(0.01)
        clf  
    end
end