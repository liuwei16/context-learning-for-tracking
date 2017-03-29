classdef yoloLoss3 < dagnn.Loss


  properties
    param = struct();  
    noobj = 2;       % weight of no object
    coord = 5.0;       % weight of coordinates
    sqrt_wh = 1;       % indicate the use of sqrt of w and h
    cls_w = 5.0;
  end

  methods
    function outputs = forward(obj, inputs, params)
      % Note: rows of prediction:  [side*side*classes][side*side*n][side*side*n*coords]
      %       rows of truth:       [1+classes+coords]*side*side  
      prediction = gather(squeeze(inputs{1})); % prediction from last layer, example : 1470*64
      truth =      gather(squeeze(inputs{2})); % truth as input, example : 1225*64 (1225 = 25*49) 
      batch =      size(prediction,2); % num of images for one computation
      global delta;
      delta = zeros(size(prediction)); % for the deviation between prediction and truth
      cost = 0;                        % for the whole cost of the loss function
      count = 0;                       % indicates how many objects in truth of the whole batch      
      for b = 1 : batch
         for i= 1 : obj.param.locations
            truth_index = (i-1)*(1+obj.param.classes+obj.param.coords);
            is_obj = logical(truth(truth_index+1,b)) ;
            for j = 1 : obj.param.n
                p_index = obj.param.locations*obj.param.classes + (i-1)*obj.param.n +j;
                delta(p_index,b) = obj.noobj*(0-prediction(p_index,b));
                cost = cost+obj.noobj*prediction(p_index,b)^2;  % the 4th term in the loss function
%                 avg_anyobj = avg_anyobj+prediction(p_index,b);
            end
           
            if ~is_obj, continue; end
            
            class_index = (i-1)*obj.param.classes;
            for j = 1 : obj.param.classes
                delta(class_index+j,b) = obj.cls_w * ( truth(truth_index+1+j,b)-prediction(class_index+j,b));
                cost = cost+delta(class_index+j,b)^2;  % the 5th term in the loss function    
            end            
            truth_box = get_box(truth(truth_index+1+obj.param.classes+1 : truth_index+1+obj.param.classes+4 , b));
            truth_box.x = truth_box.x/obj.param.side;
            truth_box.y = truth_box.y/obj.param.side;
            best_index = 0;
            best_iou = 0;
            best_rmse = 20;
            for j = 1 : obj.param.n
               box_index = obj.param.locations*(obj.param.classes + obj.param.n) + ((i-1)*obj.param.n+(j-1))*obj.param.coords; 
               pred_box = get_box(prediction(box_index+1 : box_index+4 , b));
               pred_box.x = pred_box.x/obj.param.side;
               pred_box.y = pred_box.y/obj.param.side;
               if obj.sqrt_wh, pred_box.w = pred_box.w^2; pred_box.h = pred_box.h^2; end 
               iou = get_box_iou(pred_box , truth_box);
               rmse = get_box_rmse(pred_box , truth_box);
               if best_iou > 0 || iou > 0
                    if iou > best_iou
                        best_iou = iou;
                        best_index = j;
                    end
               else
                    if(rmse < best_rmse)
                        best_rmse = rmse;
                        best_index = j;
                    end
               end               
            end
            p_index = obj.param.locations*obj.param.classes + (i-1)*obj.param.n +best_index;
            delta(p_index,b) = 1.-prediction(p_index,b);
            cost = cost - obj.noobj*prediction(p_index,b)^2;
            cost = cost + delta(p_index,b)^2;    % the 3th term in the loss function   
            cost = cost + (1-best_iou)^2;
%             avg_iou = avg_iou +best_iou;
            count = count + 1; 
         end
      end
      outputs{1} = cost;

      
      %fprintf('Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n' , avg_iou/count, avg_cat/count, avg_allcat/(count*obj.param.classes), avg_obj/count, avg_anyobj/(batch*obj.param.locations*obj.param.n), count);
      % Accumulate loss statistics.
      k = obj.numAveraged ;
      m = k + count + 1e-9 ;
      obj.average = (k * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      global delta;
      temp = 0.-reshape (delta, 1,1,size(delta,1),size(delta,2));
      derInputs = { single(temp.* derOutputs{1}*2), []} ;
      if numel(obj.param.usegpu) > 0
        derInputs = cellfun(@gpuArray, derInputs, 'uniformoutput', false) ;
      end
      derParams = {} ;
    end       

    function obj = yoloLoss3(varargin)
      obj.load(varargin) ;
      obj.loss = 'yoloLoss2';
    end
  end
end
