function vcp = vcp_bk_bsocp()
%VCP_BK_BSOCP  Yalmip Optimizer for A-SOCP.
%   vcp = VCP_BK_BSOCP(asocp) returns a struct with a YALMIP optimizer
%   object modeling B-SOCP.
%
%   The struct has fields:
%   - N [1,1] Number of discretization steps
%   - opts [] optimizer objects
%   - tf [@] function that takes solution of opt and returns t_f
%
%   The optimizer object is [vcp.opt] and it has:
%   Inputs:
%   - x_0 [4,1] initial state
%   - x_f [4,1] final state
%   - v_max [1,1] max velocity
%   - a_max [1,1] max acceleration
%   - nu [1,1] Cost penalization for duration (High nu -> low t_f)
%   - theta_p [2,N+1] \theta'(s) evaluated at s_i, i=0,...,N
%   - theta_pp [2,N+1] \theta''(s) evaluated at s_i, i=0,...,N
%   - theta_p_n [1,N+1] \|\theta'(s)\|_2 evaluated at s_i, i=0,...,N
%   - dot_norm [1,N+1] (\theta' \cdot \theta'')/\|\theta'\|_2 at s_i
%   
%
%   Outputs:
%   - a [N+1,1] a = s_ddot
%   - b [N+1,1] b = s_dot^2
%
%   Copyright 2021 Victor Freire. 

%% Setup Problem
% Discretization
vcp.N = 40;
Ds = 1/vcp.N;

% Parameters
x_0 = sdpvar(4,1); % Initial state
x_f = sdpvar(4,1); % Final state
v_max = sdpvar(1); % Max velocity
a_max = sdpvar(1); % Max acceleration
nu = sdpvar(1); % Cost weigth of t_f
theta_p = sdpvar(2,vcp.N+1);
theta_pp = sdpvar(2,vcp.N+1);
theta_p_n = sdpvar(1,vcp.N+1);
dot_norm = sdpvar(1,vcp.N+1);

% Variables
a = sdpvar(vcp.N+1,1);
b = sdpvar(vcp.N+1,1);
c = sdpvar(vcp.N+1,1);
d = sdpvar(vcp.N+1,1);

% Init problem
obj = 0;
cst = [];

%% Objective
% Minimum time
for i = 0:vcp.N-1
    obj = obj + nu*2*Ds*d(i+1);
end; clear i
% Smoothness
acc_norm = sdpvar(vcp.N+1,1);
acc_sq = sdpvar(vcp.N+1,1);
for i = 0:vcp.N
  cst = [cst, cone(a(i+1)*theta_p(:,i+1) + b(i+1)*theta_pp(:,i+1),acc_norm(i+1))];
  cst = [cst, cone([(acc_sq(i+1)-1)/sqrt(2);sqrt(2)*acc_norm(i+1)],(acc_sq(i+1)+1)/sqrt(2))];
  obj = obj + acc_sq(i+1);
end
% Start a(1) at 0
cst = [cst, a(1) == 0];

%% Constraints
% Min time obj reformulation
for i = 0:vcp.N
    cst = [cst, cone([2*c(i+1);b(i+1)-1],b(i+1)+1)];
end
for i = 0:vcp.N-1
    cst = [cst, cone([2;c(i+2)+c(i+1)-d(i+1)],c(i+2)+c(i+1)+d(i+1))];
end

% Differential cst
for i = 1:vcp.N
    cst = [cst, 2*Ds*a(i+1) == b(i+1) - b(i)];
end

% % b >= 0
% cst = [cst, b>=0];

% Initial/final velocity
cst = [cst, theta_p_n(1)^2*b(1) == x_0(3)^2];
cst = [cst, theta_p_n(end)^2*b(end) == x_f(3)^2];

% Max vel cst
cst_v = [];
for i = 0:vcp.N
    cst_v = [cst_v, theta_p_n(i+1)^2*b(i+1) <= v_max^2];
end

% Max acc cst
cst_a = [];
for i = 0:vcp.N
    cst_a = [cst_a, abs(a(i+1)*theta_p_n(i+1) + ...
        b(i+1)*dot_norm(i+1)) <= a_max];
end


%% Options
ops = sdpsettings('verbose',0,'solver','mosek','debug',0);

%% Setup Optimizers
par = {x_0,x_f,v_max,a_max,nu,theta_p,theta_pp,theta_p_n,dot_norm};
out = {a,b,acc_norm,acc_sq};

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

%% Wrapper to compute t_f
vcp.tf = @get_tf;
end

function tf = get_tf(a,b)
    s_dot = sqrt(b);
    s_ddot = a;
    N = length(s_dot)-1;
    % Recover time vector 
    t = zeros(N+1,1);
    Dt = zeros(N,1);
    for i = 1:N
        Dt(i) = (s_dot(i+1)-s_dot(i))/s_ddot(i+1);
        t(i+1) = t(i) + Dt(i);
    end
    tf = t(end);
end