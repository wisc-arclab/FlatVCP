function P_r_j = bspline_vcp(r,P,j,d,tau)
%BSPLINE_VCP j-th, r-th order Virtual Control Point
%  P_r_j = BSPLINE_VCP(r,P,j,d,tau) Returns the j-th r-th order VCP
P_r = P*bspline_bmat(r,d,tau);
P_r_j = P_r(:,j+1);