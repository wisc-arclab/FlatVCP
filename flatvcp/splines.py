#!/usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"

"""
Methods and classes to work with B-spline functions
and B-spline curves.

If you use this code in your work, please cite:

@article{freire2022optimal,
  title={Optimal Control for Kinematic Bicycle Model with
    Continuous-time Safety Guarantees: A Sequential Second-order Cone
    Programming Approach},
  author={Freire, Victor and Xu, Xiangru},
  journal={arXiv preprint arXiv:2204.08980},
  year={2022},
}


Copyright 2022 Victor Freire
"""

import numpy as np
from scipy.interpolate import BSpline as SciSpline


# Helper functions for B-Splines
class BSpline:
    def __init__(self, P, d, tau):
        self._curves = []
        self._curves.append(SciSpline(tau, np.array(P).T, d,
                                      extrapolate=False))
        for r in range(1, d+1):
            self._curves.append(self._curves[0].derivative(r))

    # This class is callable
    def __call__(self, t, r=0):
        return self._curves[r](t).T

    # Update the control points
    def update_P(self, P):
        self._curves[0].c = np.array(P).T
        for r in range(1, len(self._curves)):
            self._curves[r] = self._curves[0].derivative(r)

    # Update the knot vector
    def update_tau(self, tau):
        self._curves[0].t = np.array(tau)
        for r in range(1, len(self._curves)):
            self._curves[r] = self._curves[0].derivative(r)

    # Returns the B-spline's r-th order control points
    def get_VCP(self, r):
        inner_VCPs = self._curves[r].c.T[:, 0:s.N-r+1]
        outer_VCPs = np.zeros((s.m, r))
        return np.hstack((outer_VCPs, inner_VCPs, outer_VCPs))

    # Return the B-spline's degree

    @property
    def d(self):
        return self._curves[0].k

    # Return the B-spline's N (number of ctrl points - 1)
    @property
    def N(self):
        return self._curves[0].c.T.shape[1] - 1

    # Return the B-spline's nu (number of knots - 1)
    @property
    def nu(self):
        return len(self._curves[0].t) - 1

    # Return the B-spline curve's dimension m
    @property
    def m(self):
        return len(self._curves[0].c.T[:, 0])

    # Return the B-spline's knot vector
    @property
    def tau(self):
        return self._curves[0].t

    # Return the B-spline's control points
    @property
    def P(self):
        return self._curves[0].c.T

    # Compute clamped and uniform knot vectors
    @staticmethod
    def knots(tf, N, d):
        v = N+d+1
        tau = np.zeros((v+1))
        for i in range(d+1, N+3):
            tau[i-1] = ((i-(d+1))*tf)/(N-d+1)
        tau[N+2:] = tau[N+1]
        return tau

    # Compute B_r
    @staticmethod
    def bmat(r, d, tau):
        assert(0 <= r and r <= d),\
            "Derivative order r is out of range. 0 <= r <= d"
        N = len(tau) - d - 2
        M_d_dr = np.eye(N+1)
        for i in range(1, r+1):
            f_M = np.zeros((N-i+2, N-i+1))
            for k in range(N-i+1):
                a = (d-i+1)/(tau[k+d+1]-tau[k+i])
                f_M[k, k] = -a
                f_M[k+1, k] = a
            M_d_dr = M_d_dr@f_M
        C_r = np.hstack((np.zeros((N-r+1, r)),
                         np.eye(N-r+1),
                         np.zeros((N-r+1, r))))
        return M_d_dr@C_r

    # Compute the r-th order VCP
    @staticmethod
    def vcp(r, P, j, d, tau):
        P_r = P@BSpline.bmat(r, d, tau)
        return np.reshape(P_r[:, j], (P.shape[0], 1))\
            if np.isscalar(j) else P_r[:, j]

    # Compute \Lambda(t)
    @staticmethod
    def lamvec(t, r, d, tau):
        assert(0 <= r and r <= d),\
            "Derivative order r is out of range. 0 <= r <= d"
        N = len(tau) - d - 2
        Lam = np.zeros((N+r+1, 1 if np.isscalar(t) else len(t)))
        for i in range(r, N+1):
            Lam[i] = SciSpline.basis_element(tau[i:d-r+2+i],
                                             extrapolate=False)(t)
        Lam[-r-1][t == tau[-1]] = 1
        Lam[np.isnan(Lam)] = 0
        return Lam

    # Compute s(t)
    @staticmethod
    def curve(t, P, r, d, tau):
        assert(0 <= r and r <= d),\
            "Derivative order r is out of range. 0 <= r <= d"
        Lam_dr = BSpline.lamvec(t, r, d, tau)
        B_r = BSpline.bmat(r, d, tau)
        return P@B_r@Lam_dr


# Test the BSpline class
if __name__ == "__main__":
    import time
    # Parameters
    N = 20
    d = 4
    m = 3
    tau = BSpline.knots(1, N, d)
    P = np.random.rand(m, N+1)
    s = BSpline(P, d, tau)
    tvec = np.linspace(0, 1, 3)

    # Timing tests
    print("Timing tests:")
    t = time.time()
    s_naive = BSpline.curve(tvec, P, 0, d, tau)
    elapsed = time.time() - t
    print("Naive elapsed: ", elapsed)
    t = time.time()
    s_sci = s(tvec, 0)
    elapsed = time.time() - t
    print("SciSpline elapsed: ", elapsed)
    assert((s_naive == s_sci).all())
    P = np.random.rand(3, N+1)
    s.update_P(P)
    t = time.time()
    s_naive = BSpline.curve(tvec, P, 1, d, tau)
    elapsed = time.time() - t
    print("Naive elapsed: ", elapsed)
    t = time.time()
    s_sci = s(tvec, 1)
    elapsed = time.time() - t
    print("SciSpline elapsed: ", elapsed)
    assert((s_naive - s_sci <= 1e-6).all())

    # Property tests
    print("\nProperty tests:")
    assert(s.d == d)
    print("Test s.d: Pass")
    assert((s.tau == tau).all())
    print("Test s.tau: Pass")
    assert((s.P == P).all())
    print("Test s.P: Pass")
    assert(s.N == N)
    print("Test s.N: Pass")
    assert(s.nu == len(tau)-1)
    print("Test s.nu: Pass")
    assert(s.m == m)
    print("Test s.m: Pass")

    # VCP Test
    print("\nControl point tests:")
    for r in range(d+1):
        VCP_static = BSpline.vcp(r, P, list(range(N+1+r)), d, tau)
        VCP_method = s.get_VCP(r)
        assert((VCP_static - VCP_method <= 1e-6).all())
        print("Test get_vcp({}): Pass".format(r))
