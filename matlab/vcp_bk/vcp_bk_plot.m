function [] = vcp_bk_plot(fnum, type, data, col, x, u, t)
%VCP_BK_PLOT  Plot the trajectory according to the passed type.
%   [] = VCP_BK_PLOT(fnum, type, data, col, x, u, t) show in
%           figure(fnum). Implemented plot types are:
%               - "path" Plots the path (x,y)
%               - "state" Plot x(t), y(t), v(t), psi(t) [2,2]
%               - "input" Plot v_dot(t), psi_dot(t), gamma(t) [3,1]
%
%   Copyright (c) 2022, University of Wisconsin-Madison


switch type
  case "path"
    if(~ishandle(fnum))
      figure(fnum)
      grid on
      xlabel("$x$ [m]","Interpreter","Latex");
      ylabel("$y$ [m]","Interpreter","Latex");
    end
    figure(fnum)
    hold on
    plot(x(1,:),x(2,:),col,'LineWidth',1.5);

  case "state"
    if(~ishandle(fnum))
      figure(fnum)
      tiledlayout(2,2);
      labels = ["x","y","v","\psi"];
      units = ["m","m","m/s","deg"];
      for i = 1:4
        nexttile(i);
        xlabel("$t$ [s]","Interpreter","Latex");
        ylabel(strcat("$",labels(i),"(t)$ [", units(i), "]"),"Interpreter","Latex");
        grid on
      end
      if(data.v_max)
        nexttile(3)
        hold on
        plot([t(1),t(end)],[data.v_max, data.v_max],'--r')
      end
    end
    figure(fnum);
    for i = 1:3
      nexttile(i);
      hold on
      plot(t,x(i,:),col,'LineWidth',1.5);
    end
    nexttile(4)
    hold on
    plot(t,rad2deg(x(4,:)),col,'LineWidth',1.5);


  case "input"
    if(~ishandle(fnum))
      figure(fnum)
      tiledlayout(3,1);
      labels = ["\dot{v}","\dot{\psi}","\gamma"];
      units = ["m/s$^2$","deg/s","deg"];
      for i = 1:3
        nexttile(i);
        xlabel("$t$ [s]","Interpreter","Latex");
        ylabel(strcat("$",labels(i),"(t)$ [", units(i), "]"),"Interpreter","Latex");
        grid on
        hold on
      end
      if(data.a_max)
        nexttile(1)
        hold on
        plot([t(1),t(end)],[data.a_max, data.a_max],'--r')
        plot([t(1),t(end)],-[data.a_max, data.a_max],'--r')
      end
      if(data.gamma_max)
        nexttile(3)
        hold on
        plot([t(1),t(end)],rad2deg([data.gamma_max, data.gamma_max]),'--r')
        plot([t(1),t(end)],-rad2deg([data.gamma_max, data.gamma_max]),'--r')
      end
    end
    figure(fnum);
    nexttile(1)
    hold on
    plot(t,u(1,:),col,'LineWidth',1.5);
    for i = 2:3
      nexttile(i);
      hold on
      plot(t,rad2deg(u(i,:)),col,'LineWidth',1.5);
    end
 
  otherwise
    error(strcat("ERROR: Type ", type, " is not implemented"))
end
