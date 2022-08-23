#!/usr/bin/env python3
import matplotlib.pyplot as plt  # Plotting
import numpy as np
import os
import sys

# Add flatvcp to path
sys.path.insert(0, os.path.abspath(  # nopep8
    os.path.join(os.path.dirname(__file__), '..')))  # nopep8
from flatvcp import bike  # Import solver nopep8


# Initialize data objects with default data
data1 = bike.FlatData()
data2 = bike.FlatData()

# Set wheelbase length
data1.L = 0.1683
data2.L = 0.1683

# Modify data
data1.x_0[:, 0] = (-1.5, 1.0, 0, 0)
data1.x_f[:, 0] = (0, 0, 0.25, np.radians(-60))
data2.x_0 = data1.x_f
data2.x_f[:, 0] = (1.5, -1.0, 0, 0)

fp = bike.BicyclePlanner()  # Initialize Planner


# Solve Segment 1
fp.solve(data1)  # Solve SOCPs for data1
x1, u1, t1 = fp.full_traj()  # Recover state-space trajectory

# Solve segment 2
fp.solve(data2)  # Solve SOCPs for data2
x2, u2, t2 = fp.full_traj()  # Recover state-space trajectory

# Join segments
x = np.hstack((x1, x2))
u = np.hstack((u1, u2))
t = np.hstack((t1, t2+t1[-1]))

# Make CSV
T = np.vstack((t, x, u))
np.savetxt('traj_figS.csv', T.T, delimiter=',')

# Plot x-y trajectory
plt.plot(x[0], x[1])
plt.show()
