function [pos,target_sz,model_alphaf,model_xf] = init_kcf(im,box)
padding = 1;
lambda = 1e-4;
output_sigma_factor = 0.1;
cell_size = 4;
% interp_factor = 0.02;
kernel.sigma = 0.5;		
kernel.poly_a = 1;
kernel.poly_b = 9;
features.hog = true;
features.hog_orientations = 9;
%window size, taking padding into account
target_sz = box([7,6])-box([5,4]);
pos = box([5,4]) + floor(target_sz/2);
window_sz = floor(target_sz * (1 + padding));
%create regression labels, gaussian shaped, with a bandwidth
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
cos_window = hann(size(yf,1)) * hann(size(yf,2))';
% imshow(im);
% rectangle('Position',[box([4 5]), target_sz([2 1])],'LineWidth',4,'EdgeColor','r');
if size(im,3) > 1,
   im = rgb2gray(im);
end

%obtain a subwindow for training at newly estimated target position
patch = get_subwindow(im, pos, window_sz);
xf = fft2(get_features(patch, features, cell_size, cos_window));
kf = gaussian_correlation(xf, xf, kernel.sigma);
alphaf = yf./ (kf + lambda);   %equation for fast training£¬lambda=1e-4

model_alphaf = alphaf;
model_xf = xf;