function vcp = vcp_bk_csocp()
%VCP_BK_CSOCP  Yalmip Optimizer for C-SOCP.
%   vcp = VCP_BK_CSOCP() returns a struct with a YALMIP optimizer
%   object modeling C-SOCP.
%
%   The struct has fields:
%   - d [1,1] B-spline degree
%   - N [1,1] Number of control points (N+1)
%   - I [1,1] Integral approximation resolution
%   - r [1,1] Order of the objective function (i.e. 3 = min jerk)
%   - tau [N+d+1,1] Knot vector, clamped, uniform
%   - opt [] optimizer object
%   - opt_ucst [] optimizer object (no a_max cst)
%
%   The optimizer object is [vcp.opt] and it has:
%   Inputs:
%   - x_0 [4,1] initial state
%   - x_f [4,1] final state
%   - v_max [1,1] max velocity
%   - a_max [1,1] max acceleration
%   - ubv_th [1,1] upper bound on \|\theta'(s)\|_2
%   - uba_th [1,1] upper bound on \|\theta''(s)\|_2
%   - uba_th_root [1,1] sqrt(uba_th)
%   - BLam [N+1,I] weight for objective function
%   - B_1 [N+1,N+2] B_1 matrix
%   - B_2 [N+1,N+3] B_2 matrix
%
%   Outputs:
%   - P [1,N+1] control points
%
%   Copyright 2021 Victor Freire. 

%% Setup Problem
% Setup B-spline
vcp.d = 4; % B-spline degree
vcp.N = 20; % Control Points
vcp.I = 25; % Integral approximation resolution
vcp.r = 3; % Objective function derivative order


% Parameters
x_0 = sdpvar(4,1); % Initial state
x_f = sdpvar(4,1); % Final state
v_max = sdpvar(1); % Max velocity
a_max = sdpvar(1); % Max acceleration
ubv_th = sdpvar(1); % Upper bound on pseudo velocity
uba_th = sdpvar(1); % Upper bound on pseudo acceleration
uba_th_root = sdpvar(1); % sqrt(uba_th)
BLam = sdpvar(vcp.N+1,vcp.I,'full'); % Objective function weight
B_1 = sdpvar(vcp.N+1,vcp.N+2,'full'); % B_1 matrix
B_2 = sdpvar(vcp.N+1,vcp.N+3,'full'); % B_2 matrix

% Variables
P = sdpvar(1,vcp.N+1); % Control Points
P_1 = sdpvar(1,vcp.N+2); % 1-st order VCPs
P_2 = sdpvar(1,vcp.N+3); % 2-nd order VCPs
kappa_bar = sdpvar(vcp.N-vcp.d+1,1);
epsilon_bar = sdpvar(vcp.N-vcp.d+1,1);

% Init problem
obj = 0;
cst = [];

%% Sparsity
% Sample data with right sparsity structure
tau = bspline_knots(1,vcp.N,vcp.d);
t = linspace(0,1,vcp.I);
Lam_obj_d = bspline_lamvec(t,vcp.r,vcp.d,tau);
B_r_d = bspline_bmat(vcp.r,vcp.d,tau);
B_1_d = bspline_bmat(1,vcp.d,tau);
B_2_d = bspline_bmat(2,vcp.d,tau);
BLam_d = B_r_d*Lam_obj_d;

% Make parameters sparse
BLam = BLam.*(BLam_d ~= 0);
B_1 = B_1.*(B_1_d ~= 0);
B_2 = B_2.*(B_2_d ~= 0);

%% Objective
costvar_s = sdpvar(vcp.I,1); % i.e. \dddot{s}(t)
cst = [cst, costvar_s == (P*BLam)'];
obj = obj + costvar_s'*costvar_s;

%% Constraints
% VCPs
cst = [cst, P_1 == P*B_1];
cst = [cst, P_2 == P*B_2];
% s(0) = 0
cst = [cst, P(1) == 0];
% s(t_f) = 1
cst = [cst, P(end) == 1];
% \dot{s}(0) = v_0
j = 1; % 1-st, 1-st order VCP
cst = [cst, ubv_th*P_1(j+1) == x_0(3)];
% \dot{s}(t_f) = v_f
j = vcp.N; %N-th, 1-st order VCP
cst = [cst, ubv_th*P_1(j+1) == x_f(3)];

% \dot{s}(t) >= 0
for j = 1:vcp.N
    cst = [cst, P_1(j+1) >= 0];
end

% v_max
cst_v = [];
for j = 1:vcp.N
    cst_v = [cst_v, ubv_th*P_1(j+1) <= v_max];
end


% a_max
cst_a = [];
A_v_dot = [sqrt(2)*uba_th_root, 0; 0, -ubv_th/sqrt(2)];
b_v_dot = [0; (a_max - 1)/sqrt(2)];
c_v_dot = [0; -ubv_th/sqrt(2)];
d_v_dot = (a_max + 1)/sqrt(2);
for k = 0:vcp.N-vcp.d
  x = [kappa_bar(k+1); epsilon_bar(k+1)];
  cst_a = [cst_a, cone(A_v_dot*x + b_v_dot, c_v_dot'*x + d_v_dot)];
  for j = k+1:k+vcp.d % s_dot <= kappa_bar
    cst_a = [cst_a, P_1(j+1) <= kappa_bar(k+1)];
  end
  for j = k+2:k+vcp.d % |s_ddot| <= epsilon_bar
    cst_a = [cst_a, -epsilon_bar(k+1) <= P_2(j+1) <= epsilon_bar(k+1)];
  end
end



%% Options
ops = sdpsettings('verbose',0,'solver','mosek','debug',0);

%% Setup Optimizers
par = {x_0,x_f,v_max,a_max,ubv_th,uba_th,uba_th_root,BLam,B_1,B_2};
out = {P,kappa_bar,epsilon_bar};

% VA encoding
% V = cst_v
% A = cst_a
% -- : 0
vcp.opts{1} = optimizer(cst,obj,ops,par,out);
% -A : 1
vcp.opts{2} = optimizer([cst, cst_a],obj,ops,par,out);
% V- : 2
vcp.opts{3} = optimizer([cst, cst_v],obj,ops,par,out);
% VA : 3
vcp.opts{4} = optimizer([cst, cst_a, cst_v],obj,ops,par,out);


