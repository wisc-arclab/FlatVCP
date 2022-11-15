function data = vcp_bk_data()
%VCP_BK_DATA  Generates a default structure with
%              data for the SOCP framework.
%   data = VCP_BK_DATA() returns the struct with fields:
%       - x_0 [4,1] Initial State [m, m, m/s, rad]
%       - x_f [4,1] Final State [m, m, m/s, rad]
%       - v_max [1,1] Maximum velocity [m/s]
%       - a_max [1,1] Maximum acceleration [m/s^2]
%       - gamma_max [1,1] Maximum steering angle [rad]
%       - nu [1,1] Weight factor for t_f (high nu -> fast traj)
%       - L [1,1] Wheelbase length [m]
%
%   Copyright (c) 2022, University of Wisconsin-Madison

data.x_0 = [0;0;15;0]; % Initial state
data.x_f = [100;4;18;0]; % Final state
data.v_max = 0; % Max velocity
data.a_max = 0; % Max acceleration (0 to disable)
data.gamma_max = 0; % Max steering angle (0 to disable)
data.nu = 1; % Weight factor for t_f (high nu -> fast traj)
data.L = 2.6746; % Wheelbase length

