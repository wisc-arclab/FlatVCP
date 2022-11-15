function [sol, err] = vcp_bk_solve(socp,data)
%VCP_BK_SOLVE  Solve the Kinematic Bicycle Model Optimal Control Problem.
%   [sol,err] = VCP_BK_SOLVE(socp,data) returns t_f, the parameters of the
%                   safe B-spline path and safe B-spline speed profile.
%                   Infeasibility is described in err code as per YALMIP.
%                   The inputs are the compiled SOCP problems in the socp
%                   struct, and the problem data structure.
%
%   see vcp_bk_data() and vcp_bk_compile()
%
%   Copyright (c) 2022, University of Wisconsin-Madison

%% Options
debug = false; % Switch to true to return extra data in sol struct
%% Struct to return
sol = [];
err = [0 0 0];
sol.L = data.L; % Store the wheelbase length

%% Check data
if norm(data.x_0(1:2)-data.x_f(1:2))<=eps
    err = [1 1 1];
    return
end

if data.gamma_max >= deg2rad(90)
  error("Unsupported gamma_max")
end

%% Get VA encoding
V_act = logical(data.v_max);
A_act = logical(data.a_max);
sol.code = num2str([V_act, A_act]);
cst_code = bin2dec(sol.code)+1;

%% A-SOCP: Safe B-spline Path
x0xf = norm(data.x_0(1:2)-data.x_f(1:2));
alpha = 2*tan(data.gamma_max)/data.L*x0xf;
if data.gamma_max
  [sol_a, errorcode] = socp.a.opt(data.x_0,data.x_f,...
            x0xf,data.gamma_max,alpha,data.L);
else
  [sol_a, errorcode] = socp.a.opt_ucst(data.x_0,data.x_f,...
            x0xf,data.gamma_max,alpha,data.L);
end
% Check A-SOCP solved correctly
err(1) = errorcode;
if errorcode ~= 0 && errorcode ~= 4 && errorcode ~= -1
    return % Exit if not feasible
end
% Process the solution of A-SOCP
[sol.theta.P, ubv_th, uba_th] =  sol_a{:};
sol.theta.d = socp.a.d;
sol.theta.N = socp.a.N;
% Extra Debug Data
if debug
    sol.theta.ubv_th = ubv_th;
    sol.theta.uba_th = uba_th;
end

%% B-SOCP: Optimal t_f
% Parameters
s_line = linspace(0,1,socp.b.N+1);
theta_p = bspline_curve(s_line,sol.theta.P,1,socp.a.d,socp.a.tau);
theta_pp = bspline_curve(s_line,sol.theta.P,2,socp.a.d,socp.a.tau);
theta_p_n = vecnorm(theta_p);
dot_norm = dot(theta_p,theta_pp)./theta_p_n;

% Use appropriate optimizer
[sol_b,errorcode] = socp.b.opts{cst_code}(data.x_0, data.x_f, ...
        data.v_max, data.a_max, data.nu, ...
        theta_p, theta_pp, theta_p_n, dot_norm);
err(2) = errorcode;
if errorcode ~= 0 && errorcode ~= 4 && errorcode ~= -1
    return % Exit if not feasible
end
% Process the solution of B-SOCP
[a,b] =  sol_b{:};
sol.t_f = socp.b.tf(a,b);

%% C-SOCP: Safe B-spline Speed Profile
% Parameters
tau = bspline_knots(sol.t_f,socp.c.N,socp.c.d);
t = linspace(0,sol.t_f,socp.c.I);
Lam_obj = bspline_lamvec(t,socp.c.r,socp.c.d,tau);
B_r = bspline_bmat(socp.c.r,socp.c.d,tau);
BLam = B_r*Lam_obj;
B_1 = bspline_bmat(1,socp.c.d,tau);
B_2 = bspline_bmat(2,socp.c.d,tau);

% Use appropriate optimizer
[sol_c, errorcode] = socp.c.opts{cst_code}(data.x_0, data.x_f, ...
        data.v_max, data.a_max, ubv_th, ...
        uba_th, sqrt(uba_th), BLam, B_1, B_2);

% Store solution
[sol.s.P, kappa_bar, epsilon_bar] =  sol_c{:};

err(3) = errorcode;
if errorcode ~= 0 && errorcode ~= 4 && errorcode ~= -1
    return % Exit if not feasible
end

% Process the solution of C-SOCP
sol.s.d = socp.c.d;
sol.s.N = socp.c.N;
if debug
    sol.s.kappa_bar = kappa_bar;
    sol.s.epsilon_bar = epsilon_bar;
end
