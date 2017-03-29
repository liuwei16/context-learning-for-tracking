function [det,pos,model_alphaf,model_xf] = kcf_tracking(im,pos,target_sz,model_alphaf,model_xf)

padding = 1;
lambda = 1e-4;
output_sigma_factor = 0.1;
cell_size = 4;	
interp_factor = 0.02;
kernel.sigma = 0.5;		
kernel.poly_a = 1;
kernel.poly_b = 9;
features.hog = true;
features.hog_orientations = 9;
%window size, taking padding into account

window_sz = floor(target_sz * (1 + padding));

%create regression labels, gaussian shaped, with a bandwidth
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
cos_window = hann(size(yf,1)) * hann(size(yf,2))';
  
if size(im,3) > 1,
    im = rgb2gray(im);
end
%obtain a subwindow for detection at the position from last
%frame, and convert to Fourier domain (its size is unchanged)
patch = get_subwindow(im, pos, window_sz);
zf = fft2(get_features(patch, features, cell_size, cos_window));
kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection

conf=max(response(:));
[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
    vert_delta = vert_delta - size(zf,1);
end
if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
    horiz_delta = horiz_delta - size(zf,2);
end
pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];

%obtain a subwindow for training at newly estimated target position
patch = get_subwindow(im, pos, window_sz);
xf = fft2(get_features(patch, features, cell_size, cos_window));
kf = gaussian_correlation(xf, xf, kernel.sigma);
alphaf = yf./ (kf + lambda);   %equation for fast training£¬lambda=1e-4

%subsequent frames, interpolate model
model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;

box = [pos([2,1]) - target_sz([2,1])/2, pos([2,1]) + target_sz([2,1])/2];
box(1) = min( max(1,box(1)),319);
box(2) = min( max(1,box(2)),239);
box(3) = min( max(1,box(3)),319);
box(4) = min( max(1,box(4)),239);
det = [box conf];




end


