function vcp_bk_compile()
%VCP_BK_COMPILE  Compile YALMIP optimizer objects, save as mat file.
%
%   VCP_BK_COMPILE() creates a file called 'vcp_bk_compiled.mat' in
%          current directory. The file contains a struct with the 3
%          compiled SOCP programs: A-SOCP, B-SOCP and C-SOCP
%
%   Copyright (c) 2022 University of Wisconsin-Madison

yalmip('clear')
socp.a = vcp_bk_asocp();
socp.b = vcp_bk_bsocp();
socp.c = vcp_bk_csocp();
save("vcp_bk_compiled.mat","socp");
