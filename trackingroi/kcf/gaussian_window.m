function labels = gaussian_window( sigma, length )
  rs = ndgrid((1:length))-floor(length/2);
  labels = exp(-0.5 / (sigma*length)^2 * (rs.^2));

end

