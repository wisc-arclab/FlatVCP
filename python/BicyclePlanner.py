#! /usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"

"""
Trajectory Planner for the Bicycle Kinematic Model.

If you use this code in your work, please cite:
@article{freire2021flatness,
  title={Flatness-based Quadcopter Trajectory Planning
    and Tracking with Continuous-time Safety Guarantees},
  author={Freire, Victor and Xu, Xiangru},
  journal={arXiv preprint arXiv:2111.00951},
  year={2021}
}

Copyright 2021 Victor Freire
"""

import numpy as np
import cvxpy as cp
from BSpline import BSpline

#TODO
np.set_printoptions(suppress=True)


# Contains the parameters for the optimization problem
class FlatData:
    # Create a default instance with sample data
    def __init__(self):
        self.x_0 = np.array([0,0,15,0]).reshape([4,1])
        self.x_f = np.array([100,4,18,0]).reshape([4,1])
        self.v_max = 1000
        self.a_max = 1000
        self.gamma_max = 80
        self.nu = 1
        self.L = 2.601

# Model A-SOCP
class ASOCP:
    def __init__(self):
        # B-Spline Setup
        d = 4
        N = 20
        tau = BSpline.knots(1,N,d)
        self.theta = BSpline(np.zeros((2,N+1)),d,tau)

        # Optimization setup
        obj_order = 3
        int_res = 50
        obj = 0
        cst = []

        # Parameters
        self.r_0 = cp.Parameter((2,1))
        self.r_f = cp.Parameter((2,1))
        self.r_p_0 = cp.Parameter((2,1))
        self.r_p_f = cp.Parameter((2,1))

        # Variables
        P = cp.Variable((2,N+1))
        theta_ppp = cp.Variable((2,int_res))
        ubv_th = cp.Variable(nonneg=True)

        ## Objective
        obj += cp.sum_squares(theta_ppp.flatten())
        obj += ubv_th

        ## Constraints
        s = np.linspace(0,1,int_res)
        cst.append(theta_ppp==BSpline.curve(s,P,obj_order,d,tau))
        cst.append(BSpline.curve(0,P,0,d,tau)==self.r_0)
        cst.append(BSpline.curve(1,P,0,d,tau)==self.r_f)
        cst.append(BSpline.curve(0,P,1,d,tau)==ubv_th*self.r_p_0)
        cst.append(BSpline.curve(1,P,1,d,tau)==ubv_th*self.r_p_f)
        cst.append(cp.SOC(ubv_th*np.ones(N),
                BSpline.vcp(1,P,list(range(1,N+1)),d,tau)))


        ## Problem
        self.problem = cp.Problem(cp.Minimize(obj),cst)
        assert(self.problem.is_dcp(dpp=True))

        ## Save Variables
        self.P = P
        self.ubv_th = ubv_th

    # Solve the problem
    def _solve(self, data, solver=cp.MOSEK):
        self.r_0.value = data.x_0[0:2]
        self.r_f.value = data.x_f[0:2]
        self.r_p_0.value = np.array([np.cos(data.x_0[3]),\
                np.sin(data.x_0[3])]).reshape((2,1))
        self.r_p_f.value = np.array([np.cos(data.x_f[3]),\
                np.sin(data.x_f[3])]).reshape((2,1))
        sol = self.problem.solve(solver=solver)
        if self.problem.status == cp.OPTIMAL:
            self.theta.update_P(self.P.value)
        return sol


# Model B-SOCP
class BSOCP:
    def __init__(self):
        # Discretization setup
        N = 40
        Ds = 1/N
        self.s = np.linspace(0,1,N+1)

        # Optimization setup
        self.t_f = 0
        obj = 0
        cst = []

        # Parameters
        self.nu = cp.Parameter(nonneg=True)
        self.v_0_sq = cp.Parameter(nonneg=True)
        self.v_f_sq = cp.Parameter(nonneg=True)
        self.v_max_sq = cp.Parameter(nonneg=True)
        self.theta_p = cp.Parameter((2,N+1))
        self.theta_pp = cp.Parameter((2,N+1))
        self.theta_p_n = cp.Parameter((1,N+1), nonneg=True)
        self.theta_p_n_sq = cp.Parameter((1,N+1), nonneg=True)

        # Variables
        a = cp.Variable(N+1)
        b = cp.Variable(N+1)
        c = cp.Variable(N+1)
        d = cp.Variable(N+1)
        acc_norm = cp.Variable(N+1)
        acc_sq = cp.Variable(N+1)

        ## Objective
        obj += 2*self.nu*Ds*cp.sum(d[:-1]) + self._L(a,b)

        ## Constraints
        cst.append(cp.SOC(b+1,cp.vstack((2*c.T,b.T-1))))
        cc = c[1:] + c[:-1]
        cst.append(cp.SOC(cc+d[:-1],cp.vstack((2*np.ones(N),cc-d[:-1]))))
        cst.append(2*Ds*a[1:]==b[1:]-b[:-1])
        cst.append(b[0]*self.theta_p_n_sq[0][0]==self.v_0_sq)
        cst.append(b[-1]*self.theta_p_n_sq[0][-1]==self.v_f_sq)
        cst.append(self.theta_p_n_sq@cp.diag(b)<=self.v_max_sq)

        ## Problem
        self.problem = cp.Problem(cp.Minimize(obj),cst)
        assert(self.problem.is_dcp(dpp=True)),"Not DPP"

        ## Save Variables
        self.N = N
        self.a = a
        self.b = b

    # Lagrangian
    def _L(self,a,b):
        acc = self.theta_p@cp.diag(a) + self.theta_pp@cp.diag(b)
        return cp.sum_squares(acc.flatten())

    # Recover t_f
    def _t_f(self):
        s_dot = np.sqrt(self.b.value)
        s_ddot = self.a.value
        t = np.zeros(self.N+1)
        Dt = np.zeros(self.N)
        for i in range(self.N):
            Dt[i] = (s_dot[i+1]-s_dot[i])/s_ddot[i+1]
            t[i+1] = t[i] + Dt[i]
        return t[-1]

    # Solve the problem
    def _solve(self, data, asocp, solver=cp.MOSEK):
        self.v_0_sq.value = data.x_0[2][0]**2
        self.v_f_sq.value = data.x_f[2][0]**2
        self.v_max_sq.value = data.v_max**2
        self.nu.value = data.nu
        self.theta_p.value = asocp.theta(self.s,1)
        self.theta_pp.value = asocp.theta(self.s,2)
        self.theta_p_n.value = np.linalg.norm(self.theta_p.value,\
                axis=0).reshape(self.theta_p_n.shape)
        self.theta_p_n_sq.value = np.square(self.theta_p_n.value)
        sol = self.problem.solve(solver=solver)
        if self.problem.status == cp.OPTIMAL:
            self.t_f = self._t_f()
        else:
            self.t_f = 0
        return sol

# Model C-SOCP
class CSOCP:
    def __init__(self):
        # B-Spline Setup
        d = 4
        N = 20
        obj_order = 3
        self.s = BSpline(np.zeros((1,N+1)),d,BSpline.knots(1,N,d))

        # Optimization setup
        I = 25
        obj = 0
        cst = []

        # Parameters #TODO sparsity
        self.BLam = cp.Parameter((N+1,I))

        # Variables
        P = cp.Variable((1,N+1))
        P_1 = cp.Variable((1,N+2))
        P_2 = cp.Variable((1,N+3))
        jerk = cp.Variable((I))

        ## Objective
        obj += cp.sum_squares(jerk)

        ## Constraints
        cst.append(jerk == (P@self.BLam).flatten())

        ## Problem
        self.problem = cp.Problem(cp.Minimize(obj),cst)
        assert(self.problem.is_dcp(dpp=True)),"Not DPP"

        ## Save Variables
        self.I = I
        self.obj_order = obj_order
        self.d = d
        self.N = N
        self.P = P

    # Solve the problem
    def _solve(self, data, asocp, bsocp, solver=cp.MOSEK):
        t_f = bsocp.t_f
        t = np.linspace(0,t_f,self.I)
        tau = BSpline.knots(t_f,self.N,self.d)
        B_r = BSpline.bmat(self.obj_order,self.d,tau)
        Lam_obj = BSpline.lamvec(t,self.obj_order,self.d,tau)
        self.BLam.value = B_r@Lam_obj
        sol = self.problem.solve(solver=solver)
        if self.problem.status == cp.OPTIMAL:
            self.s.update_P(self.P.value)
            self.s.update_tau(tau)
        print(self.problem.status)
        return sol

class BicyclePlanner:
    def __init__(self):
        self.asocp = ASOCP()
        self.bsocp = BSOCP()
        self.csocp = CSOCP()

    def solve(self, data=FlatData()):
        self.asocp._solve(data)
        self.bsocp._solve(data, self.asocp)
        self.csocp._solve(data, self.asocp, self.bsocp)


# Test the planner
if __name__=="__main__":
    import matplotlib.pyplot as plt
    import time
    fp = BicyclePlanner()


    # Time the solution
    for i in range(5):
        t = time.time()
        fp.solve()
        elapsed = time.time() - t
    print("FlatPlanner elapsed: ", elapsed)

    # Plot
    t = np.linspace(0,1,50)
    theta = fp.asocp.theta(t)
    plt.plot(theta[0,:],theta[1,:])
    plt.show()

