clear; clc; close all;

%% Load compiled Yalmip optimizer
load("vcp_bk_compiled.mat","socp")

%% Load sample data
data = vcp_bk_data();
data.x_0(3) = 0;
data.x_f(3) = 0;
data.v_max = 4;
data.a_max = 0.9;
data.gamma_max = deg2rad(0.25);

%% Solve problem
[sol, err] = vcp_bk_solve(socp,data);
if(any(err))
  error("Infeasible")
end

%% Recover State-space
[x, u, t] = vcp_bk_inflate(sol, 0.01);

%% Plot
vcp_bk_plot(1,"path",data,'b',x,u,t);
vcp_bk_plot(2,"state",data,'b',x,u,t);
vcp_bk_plot(3,"input",data,'b',x,u,t);