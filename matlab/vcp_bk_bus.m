function bus = vcp_bk_bus()
%VCP_BK_BUS  Generates a Simulink.Bus object for FlatData types.
%   bus = VCP_BK_BUS() returns the Simulink.Bus object with Elements:
%       - x_0 [4,1] Initial State [m, m, m/s, rad]
%       - x_f [4,1] Final State [m, m, m/s, rad]
%       - v_max [1,1] Maximum velocity [m/s]
%       - a_max [1,1] Maximum acceleration [m/s^2]
%       - gamma_max [1,1] Maximum steering angle [rad]
%       - nu [1,1] Weight factor for t_f (high nu -> fast traj)
%       - L [1,1] Wheelbase length [m]
%   
%
%   Copyright 2022 Victor Freire. 

bus = Simulink.Bus;
bus.Description = "Flat Data";
% x_0
elems(1) = Simulink.BusElement;
elems(1).Name = "x_0";
elems(1).Dimensions = [4,1];
% x_f
elems(2) = Simulink.BusElement;
elems(2).Name = "x_f";
elems(2).Dimensions = [4,1];
% v_max
elems(3) = Simulink.BusElement;
elems(3).Name = "v_max";
% a_max
elems(4) = Simulink.BusElement;
elems(4).Name = "a_max";
% gamma_max
elems(5) = Simulink.BusElement;
elems(5).Name = "gamma_max";
% nu
elems(6) = Simulink.BusElement;
elems(6).Name = "nu";
% L
elems(7) = Simulink.BusElement;
elems(7).Name = "L";
% Add elements
bus.Elements = elems;
clear elems;