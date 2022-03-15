function vcp = vcp_bk_asocp()
%VCP_BK_ASOCP  Yalmip Optimizer for A-SOCP.
%   vcp = VCP_BK_ASOCP() returns a struct with a YALMIP optimizer
%   object modeling A-SOCP.
%
%   The struct has fields:
%   - d [1,1] B-spline degree
%   - N [1,1] Number of control points (N+1)
%   - tau [N+d+1,1] Knot vector, clamped, uniform
%   - L [1,1] wheelbase length of vehicle
%   - opt_ucst [] optimizer object (no gamma_max cst)
%   - opt [] optimizer object
%
%   The optimizer object is [vcp.opt] and it has:
%   Inputs:
%   - x_0 [4,1] initial state
%   - x_f [4,1] final state
%   - distance [1,1] \|r_f - r_0\|_2
%   - gamma_max [1,1] maximum steering angle
%   - alpha [1,1] slope for ncvx relaxation
%   - L [1,1] wheelbase length
%
%   Outputs:
%   - P [2,N+1] control points
%   - ubv_th [1,1] upper bound on \|\theta'(s)\|_2
%   - uba_th [1,1] upper bound on \|\theta''(s)\|_2
%
%   Copyright 2021 Victor Freire. 

%% Setup Problem
% Setup B-spline
vcp.d = 4;
vcp.N = 20;
vcp.tau = bspline_knots(1,vcp.N,vcp.d);
obj_order = 3;
int_res = 50; % Integral approximation resolution

% Parameters
x_0 = sdpvar(4,1); % Initial state
x_f = sdpvar(4,1); % Final state
distance = sdpvar(1); % Distance between r_0 and r_f
gamma_max = sdpvar(1); % [rad] Maximum steering angle
alpha = sdpvar(1); % Slope of ncvx relaxation (gamma_max)
L = sdpvar(1); % Wheelbase length of vehicle


% Variables
P = sdpvar(2,vcp.N+1); % Control Points
ubv_th = sdpvar(1); % Bound on \|\theta'(s)\|_2 <= ubv_th
lbv_th = sdpvar(1);
uba_th = sdpvar(1); % Bound on \|\theta''(s)\|_2 <= uba_th
beta = sdpvar(1);


% Init problem
obj = 0;
cst = [];

%% Objective
% Norm upper bounds
obj = obj + ubv_th;
obj = obj + uba_th;
obj = obj - lbv_th;
% Smoothness
t_sig_obj =  linspace(vcp.tau(1),vcp.tau(end),int_res)'; % approx integral
costvar_sig = sdpvar(2,length(t_sig_obj)); % i.e. pseudo acc of \sigma(s)
for k = 1:length(t_sig_obj)
  cst = [cst, costvar_sig(:,k) == bspline_curve(t_sig_obj(k),P,obj_order,vcp.d,vcp.tau)];
  obj = obj + costvar_sig(:,k)'*costvar_sig(:,k); % \|sig''(s)\|^2;
end

%% Constraints
% \theta(0) = r_0
cst = [cst, bspline_curve(0,P,0,vcp.d,vcp.tau) == x_0(1:2)];
% \theta(1) = r_f
cst = [cst, bspline_curve(1,P,0,vcp.d,vcp.tau) == x_f(1:2)];
% \theta'(0) = ubv_th*[cos\\sin]
cst = [cst, bspline_curve(0,P,1,vcp.d,vcp.tau) == ubv_th*[cos(x_0(4));sin(x_0(4))]];
% \theta'(1) = ubv_th*[cos\\sin]
cst = [cst, bspline_curve(1,P,1,vcp.d,vcp.tau) == ubv_th*[cos(x_f(4));sin(x_f(4))]];

% \|\theta'\|_2 <= ubv_th
r = 1;
for j = r:vcp.N
  cst = [cst, cone(bspline_vcp(r,P,j,vcp.d,vcp.tau),ubv_th)];
end

% \|\theta'\|_2 >= lbv_th
cst = [cst, lbv_th >= 0];
for j = 1:vcp.N
    d_hat = (x_f(1:2)-x_0(1:2))/distance;
    cst = [cst, d_hat'*bspline_vcp(1,P,j,vcp.d,vcp.tau) >= lbv_th];
end

% \|\theta''\|_2 <= uba_th
r = 2;
for j = r:vcp.N
  cst = [cst, cone(bspline_vcp(r,P,j,vcp.d,vcp.tau),uba_th)];
end

% |\gamma| <= gamma_max
cst_gamma = [];
cst_gamma = [cst_gamma, beta >=0];
gamma_tilde = tan(gamma_max)/L;
cst_gamma = [cst_gamma, cone([2*alpha; 4*gamma_tilde*beta-1],... 
       4*gamma_tilde*beta + 1)];
cst_gamma = [cst_gamma, uba_th <= alpha*lbv_th - beta];


%% Options
opt = sdpsettings('verbose',0,'solver','mosek','debug',0);

%% Setup Optimizer
% Unconstraned gamma max
vcp.opt_ucst = optimizer(cst,obj,opt,{x_0,x_f,distance,gamma_max,alpha,L},{P, ubv_th, uba_th});
% Constrained gamma max
vcp.opt = optimizer([cst, cst_gamma],obj,opt,{x_0,x_f,distance,gamma_max,alpha,L},{P, ubv_th, uba_th});
