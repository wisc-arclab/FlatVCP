#! /usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"

"""
Methods and classes to work with B-spline functions
and B-spline curves.

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
from scipy.interpolate import BSpline as SciSpline


# Helper functions for B-Splines
class BSpline:
    def __init__(self, P, d, tau):
        self._curves = []
        self._curves.append(SciSpline(tau,np.array(P).T,d,\
                extrapolate=False))
        for r in range(1,d+1):
            self._curves.append(self._curves[0].derivative(r))

    def __call__(self,t,r=0):
        return self._curves[r](t).T

    def update_P(self, P):
        self._curves[0].c = np.array(P).T
        for r in range(1,len(self._curves)):
            self._curves[r] = self._curves[0].derivative(r)

    def update_tau(self, tau):
        self._curves[0].t = np.array(tau)
        for r in range(1,len(self._curves)):
            self._curves[r] = self._curves[0].derivative(r)

    @staticmethod
    def knots(tf,N,d):
        v = N+d+1
        tau = np.zeros((v+1))
        for i in range(d+1,N+3):
            tau[i-1] = ((i-(d+1))*tf)/(N-d+1);
        tau[N+2:] = tau[N+1];
        return tau

    @staticmethod
    def bmat(r,d,tau):
        assert(0<= r and r<=d),\
                "Derivative order r is out of range. 0 <= r <= d"
        N = len(tau) - d - 2
        M_d_dr = np.eye(N+1)
        for i in range(1,r+1):
            f_M = np.zeros((N-i+2,N-i+1))
            for k in range(N-i+1):
                a = (d-i+1)/(tau[k+d+1]-tau[k+i])
                f_M[k,k]= -a
                f_M[k+1,k] = a
            M_d_dr = M_d_dr@f_M
        C_r = np.hstack((np.zeros((N-r+1,r)),\
                np.eye(N-r+1),\
                np.zeros((N-r+1,r))))
        return M_d_dr@C_r

    @staticmethod
    def vcp(r,P,j,d,tau):
        P_r = P@BSpline.bmat(r,d,tau)
        return np.reshape(P_r[:,j],(P.shape[0],1))\
            if np.isscalar(j) else P_r[:,j]

    @staticmethod
    def lamvec(t,r,d,tau):
        assert(0<= r and r<=d),\
                "Derivative order r is out of range. 0 <= r <= d"
        N = len(tau) - d - 2;
        Lam = np.zeros((N+r+1, 1 if np.isscalar(t) else len(t)))
        for i in range(r,N+1):
            Lam[i] = SciSpline.basis_element(tau[i:d-r+2+i],\
                    extrapolate=False)(t)
        Lam[-r-1][t==tau[-1]] = 1
        Lam[np.isnan(Lam)] = 0
        return Lam

    @staticmethod
    def curve(t,P,r,d,tau):
        assert(0<= r and r<=d),\
                "Derivative order r is out of range. 0 <= r <= d"
        Lam_dr = BSpline.lamvec(t,r,d,tau)
        B_r = BSpline.bmat(r,d,tau)
        return P@B_r@Lam_dr




# Test the planner
if __name__=="__main__":
    import time
    N = 20
    d = 4
    tau = BSpline.knots(1,N,d)
    #print(tau)
    P = np.random.rand(3,N+1)
    s = BSpline(P,d,tau)
    #P = (0,1,2)
    #print(P)
    #print(BSpline.lamvec((1,0.5,0),0,d,tau))
    #print(BSpline.lamvec((0,0.5,1),2,d,tau))

    tvec = np.linspace(0,1,3)

    # Time Curves Naive
    t = time.time()
    print(BSpline.curve(tvec,P,0,d,tau))
    elapsed = time.time() - t
    print("BSpline elapsed: ", elapsed)

    # Time Curves SciSpline
    t = time.time()
    print(s(tvec,0))
    elapsed = time.time() - t
    print("SciSpline elapsed: ", elapsed)

    # Update P
    P = np.random.rand(3,N+1)
    s.update_P(P)

    # Time Curves Naive
    t = time.time()
    print(BSpline.curve(tvec,P,1,d,tau))
    elapsed = time.time() - t
    print("BSpline elapsed: ", elapsed)

    # Time Curves SciSpline
    t = time.time()
    print(s(tvec,1))
    elapsed = time.time() - t
    print("SciSpline elapsed: ", elapsed)

    # VCP Test
    #print(P)
    #print(BSpline.vcp(0,P,0,d,tau))
    #print(BSpline.vcp(0,P,(0,1,2,3,4,5),d,tau))
