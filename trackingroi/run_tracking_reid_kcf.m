function run_tracking_reid_kcf(varargin)
%run_tracking_reid demonstrate the tracking with person reid
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..','externel', 'matconvnet','matlab', 'vl_setupnn.m')) ;
addpath(genpath('./'))
init_tracker
load('id2.mat');load('id3.mat');
id3=[id3;id2];
for k =1:size(id3,1)  % indicate the the ID of a single target(person)
% for idp = 0: 1: numP  % indicate the the ID of a single target(person)
    idp = id3(k);
    resultPath=['./results/' num2str(idp) '.mat'];
    if exist(resultPath,'file')
       continue; 
    end
    % init the first frame of target of interest(TOI)
    ids      = find(det(:, 3) == idp);  
    active_c = det(ids(1), 1);    % active camera
    start_f  = det(ids(1), 2)+1;  % starting frame, as the index of frame in det is from 0
    end_f = det(ids(end), 2)+1;
    traj     = [det(ids(1),:) 1];% for the whole trajectory of the target, given the first bbx with confidence 1
    traj(6)=traj(4)+traj(6); traj(7)=traj(5)+traj(7);
    % first frame fine tuning    
    fnet = first_frame_FT(pnet,active_c,start_f,idp,opts);    
    % extract probeim and roipooling feature
    
    fstim = imread([opts.v_path, sprintf('/images/%s%05d.jpg',num2str(active_c),start_f)]);
    probeim = imresize(single(fstim),[455 455],'Method','bilinear');
    scale = 455./[320 240];
    box(:,[1 3])= traj(:,[4 6])*scale(1);
    box(:,[2 4])= traj(:,[5 7])*scale(2);
    roi = single([1 ; box']) ;
    roinet.eval({'input', probeim, 'rois', roi});
    proberp = squeeze(gather(roinet.vars(roinet.getVarIndex('xRP')).value)) ;
    % init kcf
    [pos,target_sz,model_alphaf,model_xf] = init_kcf(fstim,traj);

    % go tracking from the 2nd frame
    idf = start_f;
    flag_single = 1;
    update      = 1;
    while idf < end_f
        if flag_single
            frame = imread([opts.v_path, sprintf('/images/%s%05d.jpg',num2str(active_c),idf+1)]);
    %         detection= perform_detection(frame,fnet,opts,idf,idp,active_c,0.2);% bbx is x,y of two edges
            %----kcftracking----
            tic
            [track,pos,model_alphaf,model_xf] = kcf_tracking(frame,pos,target_sz,model_alphaf,model_xf);
            time=toc;
            if track(end)>0.3
                res = [active_c,idf,idp,track];
            else
                flag_single = 0;
%                 update      = 1;
                res = []; 
            end        
            %-------------------          
        else
            if update, fnet = inter_camera_FT(fnet,idf,traj,opts);end
            idf = idf + opts.step; if idf>end_f, break;end            
            for i = 1:3, frame(:,:,:,i) =  imread([opts.v_path, sprintf('/images/%s%05d.jpg',num2str(i),idf+1)]); end
            detection = perform_detection(frame,fnet,opts,idf,idp,active_c,0.5);
            recognition = personreid(proberp,frame,detection,roinet,rnet,opts);
            if ~isempty(recognition)             
                res = recognition;
                flag_single = 1;  
                active_c    = res(1);
                update = 0;
                [pos,target_sz,model_alphaf,model_xf] = init_kcf(frame(:,:,:,active_c),res);
            else                 
                res    = [];
                update = 0;
            end
        end        
        traj = [traj; res];
        idf  = idf + 1;
        clear frame;
        disp_results
    end
    fprintf('id%s: done\n', num2str(idp));
    clear fnet;
    
    save(resultPath,'traj');  % save results for single person
end

