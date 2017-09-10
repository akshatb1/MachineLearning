function y= normal(x,mean,covariance)
d = max(size(x));
y=  (exp(-0.5*(x-mean)'*inv(covariance)*(x-mean)))/(((2*pi)^d/2)*(det(covariance)^0.5));
