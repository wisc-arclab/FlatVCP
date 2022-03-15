#!/usr/bin/env python3

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

#np.set_printoptions(suppress=True)


# Contains the parameters for the optimization problem
class FlatData:
    # Create a default instance with sample data
    def __init__(self):
        self.x_0 = np.array([0,0,15,0]).reshape([4,1])
        self.x_f = np.array([100,4,18,0]).reshape([4,1])
        self.v_max = 1000 # Max velocity
        self.a_max = 1000 # Max acceleration (0 to disable)
        self.gamma_max = 0 # Max steering angle (0 to disable)
        self.nu = 1 # Time penalty (high nu -> high speed)
        self.L = 2.601 # Wheelbase length

# Model PATH-SOCP
class PATH_SOCP:
    def __init__(self):
        # B-Spline Setup
        d = 4 # B-spline degree
        N = 20 # Number of control points
        tau = BSpline.knots(1,N,d) # Knot vector
        self.theta = BSpline(np.zeros((2,N+1)),d,tau) # B-spline curve

        # Optimization setup
        obj_order = 3 # 3 -> min curvature, 1 -> min length
        int_res = 50  # Integral approximation resolution
        obj = 0 # General objective
        obj_gamma = 0 # Added objective for gamma_max
        cst = [] # General constraints
        cst_gamma = [] # Constraints for gamma_max (conservative)

        # Parameters
        self.r_0 = cp.Parameter((2,1))   # Initial position
        self.r_f = cp.Parameter((2,1))   # Final position
        self.r_p_0 = cp.Parameter((2,1)) # Initial pseudo vel unit vec
        self.r_p_f = cp.Parameter((2,1)) # Final pseudo vel unit vec
        self.d_hat = cp.Parameter((2,1)) # Heuristic for gamma_max
        self.alpha = cp.Parameter(nonneg=True) # Heuristic for gamma_max
        self.gamma_tilde = cp.Parameter(nonneg=True) # tan(gamma_max)/L

        # Variables
        P = cp.Variable((2,N+1)) # Control points
        theta_obj = cp.Variable((2,int_res)) # theta^{(obj_order)}
        ubv_th = cp.Variable(nonneg=True)    # \overline{v}_{\theta}
        lbv_th = cp.Variable(nonneg=True)    # \underline{v}_{\theta}
        uba_th = cp.Variable(nonneg=True)    # \overline{a}_{\theta}
        beta = cp.Variable(nonneg=True)      # Ncvx relaxation for gamma_max

        ## Objective
        obj += cp.sum_squares(theta_obj.flatten())
        obj += ubv_th # Minimize
        obj += uba_th # Minimize
        obj_gamma -= lbv_th # Maximize

        ## Constraints
        # Approximate obj integral
        s = np.linspace(0,1,int_res)
        cst.append(theta_obj==BSpline.curve(s,P,obj_order,d,tau))
        # Initial and final position
        cst.append(BSpline.curve(0,P,0,d,tau)==self.r_0)
        cst.append(BSpline.curve(1,P,0,d,tau)==self.r_f)
        # Initial and final heading
        cst.append(BSpline.curve(0,P,1,d,tau)==ubv_th*self.r_p_0)
        cst.append(BSpline.curve(1,P,1,d,tau)==ubv_th*self.r_p_f)
        # \|\theta'(s)\|<= \overline{v}_{\theta}
        cst.append(cp.SOC(ubv_th*np.ones(N),
                BSpline.vcp(1,P,list(range(1,N+1)),d,tau)))
        # gamma(s) <= gamma_max
        cst_gamma.append(self.d_hat.T@BSpline.vcp(1,P,
                list(range(1,N+1)),d,tau) >= lbv_th)
        cst_gamma.append(cp.SOC(uba_th*np.ones(N-1),
                BSpline.vcp(2,P,list(range(2,N+1)),d,tau)))
        cst_gamma.append(cp.SOC(4*self.gamma_tilde*beta + 1,
                cp.vstack((2*self.alpha, 4*self.gamma_tilde*beta-1))))
        cst_gamma.append(uba_th <= self.alpha*lbv_th - beta)

        ## Problems
        self.problem_ucst = cp.Problem(cp.Minimize(obj),cst)
        assert(self.problem_ucst.is_dcp(dpp=True))
        self.problem = cp.Problem(cp.Minimize(obj + obj_gamma),
                cst + cst_gamma)
        assert(self.problem.is_dcp(dpp=True))

        ## Save Variables
        self.P = P
        self.ubv_th = ubv_th
        self.lbv_th = lbv_th
        self.uba_th = uba_th

    # Solve the problem
    def _solve(self, data, solver):
        # Populate general parameters
        self.r_0.value = data.x_0[0:2]
        self.r_f.value = data.x_f[0:2]
        self.r_p_0.value = np.array([np.cos(data.x_0[3]),\
                np.sin(data.x_0[3])]).reshape((2,1))
        self.r_p_f.value = np.array([np.cos(data.x_f[3]),\
                np.sin(data.x_f[3])]).reshape((2,1))
        # Check if gamma_max is required
        if data.gamma_max:
            delta = data.x_f[0:2]-data.x_0[0:2]
            self.d_hat.value = delta/np.linalg.norm(delta)
            self.gamma_tilde.value = np.tan(data.gamma_max)/data.L
            self.alpha.value = 2*np.tan(data.gamma_max)\
                    /data.L*np.linalg.norm(delta)
            sol = self.problem.solve(solver=solver)
        else:
            sol = self.problem_ucst.solve(solver=solver)
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
        self.a_max = cp.Parameter(nonneg=True)
        self.theta_p = cp.Parameter((2,N+1))
        self.theta_pp = cp.Parameter((2,N+1))
        self.theta_p_n = cp.Parameter((1,N+1), nonneg=True)
        self.theta_p_n_sq = cp.Parameter((1,N+1), nonneg=True)
        self.dot_norm = cp.Parameter((1,N+1))

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
        cst.append(a[0] == 0)
        cst.append(cp.abs(self.theta_p_n@cp.diag(a) +
                self.dot_norm@cp.diag(b)) <= self.a_max)

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
    def _solve(self, data, psocp, solver=cp.MOSEK):
        self.v_0_sq.value = data.x_0[2][0]**2
        self.v_f_sq.value = data.x_f[2][0]**2
        self.v_max_sq.value = data.v_max**2
        self.a_max.value = data.a_max
        self.nu.value = data.nu
        self.theta_p.value = psocp.theta(self.s,1)
        print(psocp.theta.c)
        self.theta_pp.value = psocp.theta(self.s,2)
        self.theta_p_n.value = np.linalg.norm(self.theta_p.value,\
                axis=0).reshape(self.theta_p_n.shape)
        self.theta_p_n_sq.value = np.square(self.theta_p_n.value)
        self.dot_norm.value = np.einsum('ij,ij->j',
                self.theta_p.value,self.theta_pp.value)\
                        /self.theta_p_n.value
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

        # Sparsity pattern
        tau = BSpline.knots(1,N,d);
        t = np.linspace(0,1,I);
        Lam_obj_d = BSpline.lamvec(t,obj_order,d,tau);
        B_r_d = BSpline.bmat(obj_order,d,tau);
        B_1_d = BSpline.bmat(1,d,tau);
        B_2_d = BSpline.bmat(2,d,tau);
        BLam_d = B_r_d@Lam_obj_d

        # Parameters #TODO sparsity
        self.v_0 = cp.Parameter(nonneg=True)
        self.v_f = cp.Parameter(nonneg=True)
        self.v_max = cp.Parameter(nonneg=True)
        self.a_max = cp.Parameter(nonneg=True)
        self.ubv_th = cp.Parameter(nonneg=True)
        self.uba_th_root = cp.Parameter(nonneg=True)
        self.BLam = cp.Parameter((N+1,I),sparsity=True)
        self.B_1 = cp.Parameter((N+1,N+2),sparsity=True)
        self.B_2 = cp.Parameter((N+1,N+3),sparsity=True)

        # Variables
        jerk = cp.Variable((I))
        P = cp.Variable((1,N+1))
        P_1 = cp.Variable((1,N+2))
        P_2 = cp.Variable((1,N+3))
        kappa_bar = cp.Variable(N-d+1, nonneg=True)
        epsilon_bar = cp.Variable(N-d+1, nonneg=True);

        ## Objective
        obj += cp.sum_squares(jerk)

        ## Constraints
        cst.append(jerk == (P@self.BLam).flatten())
        cst.append(P_1 == P@self.B_1)
        cst.append(P_2 == P@self.B_2)
        cst.append(P[0][0] == 0)
        cst.append(P[0][-1] == 1)
        cst.append(self.ubv_th*P_1[0][1] == self.v_0)
        cst.append(self.ubv_th*P_1[0][N] == self.v_f)
        cst.append(P_1[0][1:-1] >= 0)
        cst.append(self.ubv_th*P_1[0][1:N+1] <= self.v_max)
        A_v_dot = cp.diag(cp.vstack((np.sqrt(2)*self.uba_th_root,
                        -self.ubv_th/np.sqrt(2))))
        b_v_dot = cp.vstack((0,(self.a_max - 1)/np.sqrt(2)))
        c_v_dot = cp.vstack((0,-self.ubv_th/np.sqrt(2)))
        d_v_dot = (self.a_max + 1)/np.sqrt(2)
        x = cp.vstack((kappa_bar,epsilon_bar))
        cst.append(cp.SOC((c_v_dot.T@x).flatten() + d_v_dot,
                A_v_dot@x + b_v_dot))
        for k in range(N-d):
            cst.append(P_1[0][k+1:k+d] <= kappa_bar[k])
            cst.append(cp.abs(P_2[0][k+2:k+d]) <= epsilon_bar[k])

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
    def _solve(self, data, psocp, bsocp, solver=cp.MOSEK):
        t_f = bsocp.t_f
        t = np.linspace(0,t_f,self.I)
        tau = BSpline.knots(t_f,self.N,self.d)
        B_r = BSpline.bmat(self.obj_order,self.d,tau)
        B_1 = BSpline.bmat(1,self.d,tau)
        B_2 = BSpline.bmat(2,self.d,tau)
        Lam_obj = BSpline.lamvec(t,self.obj_order,self.d,tau)
        self.BLam.value = B_r@Lam_obj
        self.B_1.value = B_1
        self.B_2.value = B_2
        self.v_0.value = data.x_0[2][0]
        self.v_f.value = data.x_f[2][0]
        self.v_max.value = data.v_max
        self.a_max.value = data.a_max
        self.ubv_th.value = psocp.ubv_th.value
        self.uba_th_root.value = np.sqrt(psocp.uba_th.value)
        sol = self.problem.solve(solver=solver)
        if self.problem.status == cp.OPTIMAL:
            self.s.update_P(self.P.value)
            self.s.update_tau(tau)
        return sol

class BicyclePlanner:
    def __init__(self, solver=cp.MOSEK):
        # Store chosen solver
        self.solver = solver
        # Parameters
        self.Ts = 0.01 # Resolution for the trajectory
        self.data = FlatData()
        # Compile solvers
        self.psocp = PATH_SOCP()
        self.bsocp = BSOCP()
        self.csocp = CSOCP()

    def solve(self, data=FlatData()):
        self.psocp._solve(data, self.solver)
        print(self.psocp)
        self.bsocp._solve(data, self.psocp)
        self.csocp._solve(data, self.psocp, self.bsocp)
        self.data = data

    def full_traj(self):
        theta = self.psocp.theta
        s = self.csocp.s
        t = np.arange(0,self.bsocp.t_f,self.Ts)
        x = np.zeros((4,len(t)))
        u = np.zeros((3,len(t)))
        s_v = s(t).flatten()
        s_dv = s(t,1).flatten()
        s_ddv = s(t,2).flatten()
        theta_pv = theta(s_v,1)
        theta_ppv = theta(s_v,2)
        theta_pv_n = np.linalg.norm(theta_pv,axis=0)
        x[0:2] = theta(s_v)
        x[2] = theta_pv_n*s_dv
        x[3] = np.arctan2(theta_pv[1],theta_pv[0])
        psi_p = (theta_ppv[1]*theta_pv[0] -
                theta_ppv[0]*theta_pv[1])/theta_pv_n**2
        u[0] = s_ddv*theta_pv_n + (s_dv**2)*\
                np.einsum('ij,ij->j',theta_pv,theta_ppv)/theta_pv_n
        u[1] = s_dv*psi_p
        u[2] = np.arctan2(self.data.L*psi_p,theta_pv_n)
        return x, u, t


# Test the planner
if __name__=="__main__":
    import matplotlib.pyplot as plt
    import time
    fp = BicyclePlanner()

    data = FlatData()
    #data.x_f[0] = 20
    #data.x_f[1] = 20
    data.x_0[2] = 0.2
    data.x_f[2] = 5
    data.a_max = 0.4
    #data.gamma_max = np.radians(30)

    # Time the solution
    for i in range(20):
        t = time.time()
        fp.solve(data)
        elapsed = time.time() - t
    print("FlatPlanner elapsed: ", elapsed)
    x, u, t = fp.full_traj()


    # Plot
    #t = np.linspace(0,1,50)
    #theta = fp.psocp.theta(t)
    #plt.plot(t,np.degrees(u[0]))
    plt.plot(x[0],x[1])
    #plt.plot(t,u[0])
    plt.show()

