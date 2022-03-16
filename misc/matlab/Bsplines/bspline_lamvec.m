function Lambda = bspline_lamvec(t,r,d,tau)
%BSPLINE_LAMVEC d-degree B-spline basis vector (r-order-elevation)
%evaluated at t
%  s = BSPLINE_LAMVEC(t,r,d,tau) returns the B-spline basis vector
%  evaluated at t.
Lambda = bspline_basismatrix(d+1-r,tau,t)';