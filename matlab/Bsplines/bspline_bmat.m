function B_r = bspline_bmat(r,d,tau)
%BSPLINE_BMAT  Degree elevation matrix B_r to calculate Virtual Control Points
%  B_r = BSPLINE_BMAT(r,d,tau) returns the matrix B_r for a d-degree B-spline
%  over knot vector tau (clamped, uniform).
assert(0<= r && r<=d,"Derivative order r is out of range. EXPECTED: 0 <= r <= d")
N = length(tau) - d - 2;
% Build M Matrix
M_d_dr = eye(N+1);
for i = 1:r
    % Build f_M
    f_M = zeros(N-i+2,N-i+1);
    for k = 0:N-i
      a = (d-i+1)/(tau(k+d+1+1)-tau(k+i+1));
      f_M(k+1,k+1)= -a;
      f_M(k+2,k+1) = a;
    end
    % Update M_d_dr
    M_d_dr = M_d_dr*f_M;
end
% Build C Matrix
C_r = zeros(N-r+1,N+r+1);
C_r(:,r+1:end-r) = eye(N+1-r);
% Build B Matrix
B_r = M_d_dr*C_r;