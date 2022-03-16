function tau = bspline_knots(tf,N,d)
%BSPLINE_KNOTS  Clamped and Uniform knot vector for B-splines
%  tau = BSPLINE_KNOTS(tf,N,d) returns the clamped, uniform knot
%  vector segmenting the interval [0,tf]. The length of the knot vector is 
%  given by nu = N + d + 1.
v = N+d+1;
tau = zeros(v+1,1);
for i = d+1:N+2
    tau(i) = ((i-(d+1))*tf)/(N-d+1);
end
tau(N+3:end) = tau(N+2);