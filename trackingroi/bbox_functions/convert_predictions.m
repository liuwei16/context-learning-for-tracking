function [boxes, probs] = convert_predictions(prediction, param)

boxes = zeros(param.coords,param.locations*param.n);
probs = zeros(1,param.locations*param.n);
for i = 1 : param.locations
    row = ceil(i / param.side) - 1;
    col = i - row*param.side - 1;
    for j = 1 : param.n
       index = (i-1)*param.n+j-1;
       probs(1,index+1) = prediction(index+1);
       box_index = param.locations*param.n + index*param.coords;
       boxes(1,index+1) = (prediction(box_index+1)+col)/param.side;
       boxes(2,index+1) = (prediction(box_index+2)+row)/param.side;
       boxes(3,index+1) = (prediction(box_index+3))^2;
       boxes(4,index+1) = (prediction(box_index+4))^2;
    end
    
end