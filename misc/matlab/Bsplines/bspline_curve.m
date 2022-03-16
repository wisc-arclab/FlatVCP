function s = bspline_curve(t,P,r,d,tau)
%BSPLINE_CURVE d-degree B-spline curve with control points P evaluated at t
%  s = BSPLINE_CURVE(t,P,r,d,tau) returns the B-spline curve evaluated at
%  t. The control points are P. Returns the r-th order derivative.
Lam_dr = bspline_basismatrix(d+1-r,tau,t)';
s = P*bspline_bmat(r,d,tau)*Lam_dr;