'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2023, Alexandre Didier, Jérôme Sieber, Rahel Rickenbach and Shao (Mike) Zhang, ETH Zurich,
% {adidier,jsieber, rrahel}@ethz.ch
%
% All rights reserved.
%
% This code is only made available for students taking the advanced MPC 
% class in the fall semester of 2023 (151-0371-00L) and is NOT to be 
% distributed.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

import numpy as np
import cvxpy as cp
from numpy.linalg import matrix_power
from scipy.stats.distributions import chi2
from scipy.linalg import sqrtm

from ampyc.utils import Polytope, _reduce

def _robust_pre_set(Omega, A, W):
    '''
    Compute the robust pre-set of the polytopic set Omega under the linear
    autonomous dynamics A and polytopic disturbance set W.
    '''
    b_pre = Omega.b.copy()
    for i in range(Omega.b.shape[0]):
        b_pre[i] -= W.support(Omega.A[i,:])

    return Polytope(A=Omega.A @ A, b=b_pre)


def compute_mrpi(A, Omega, W, max_iter=50):
    '''
    Compute the maximal robust positive invariant set of the polytopic set Omega
    under the linear autonomous dynamics A.
    '''
    iters = 0
    mrpi = Polytope(A=Omega.A, b=Omega.b)

    while iters < max_iter:
        iters += 1
        mrpi_pre = _robust_pre_set(mrpi, A, W)
        mrpi_next = mrpi.intersect(mrpi_pre)

        if mrpi == mrpi_next:
            print('MRPI computation converged after {0} iterations.'.format(iters))
            break

        if iters == max_iter:
            print('MRPI computation did not converge after {0} max iterations.'.format(iters))
            break

        mrpi = mrpi_next

    return _reduce(mrpi)

def compute_drs(A_BK:np.array, W:Polytope, N:int) -> list:
    '''
    Compute the Disturbance Reachable Set (DRS) of the disturbance set W
    propagated by the closed-loop dynamics A_BK.
    '''
    F = (N+1) * [None]
    F[0] = Polytope() # F_0 as an empty polytope
    F[1] = W
    for i in range(1, N):
        F[i+1] = F[i] + matrix_power(A_BK, i) @ W
    return F

def compute_prs(sys, p, N):
    '''
    Computes PRS sets and the corresponding tightening using an optimization problem.

    Returns list of x_tight and u_tight with the constraint tightenings
    for time steps 1 to N and terminal tightening at time step N+1 and
    a cell P where the PRS Sets for time steps 1 to N are given
    as well as the Chebyshev Reachable Set for all time steps
    at index N+1
    '''

    # look up system parameters
    n = sys.n
    m = sys.m
    noise_cov = sys.noise_generator.cov

    # dynamics matrices
    A = sys.A
    B = sys.B

    # compute p_tilde
    p_tilde = chi2.ppf(p, n)
    sqrt_p_tilde = np.sqrt(p_tilde)

    # compute tightening according to SDP
    E = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))

    objective = cp.Minimize(cp.trace(E))

    constraints = []
    constraints += [E >> 0]
    constraints += [cp.bmat([[-noise_cov + E, (A @ E + B @ Y)],
                             [(A @ E + B @ Y).T, E]]) >> 0]

    cp.Problem(objective, constraints).solve()

    # extract tube controller
    P = np.linalg.inv(np.array(E.value))
    K = np.array(Y.value) @ P
    A_K = A + B @ K

    # error variance
    var_e = (N+1) * [None]
    var_e[0] = np.zeros((n,n))
    for i in range(N):
        var_e[i+1] = A_K @ var_e[i] @ A_K.T + noise_cov

    # set F
    F = (N+1) * [None]
    for i in range(N):
        F[i] = np.linalg.inv(var_e[i+1])
    F[-1] = P

    # compute tightening
    X = sys.X
    U = sys.U
    nx = X.A.shape[0]
    nu = U.A.shape[0]

    x_tight = np.zeros((nx,N+1))
    u_tight = np.zeros((nu,N+1))

    # for every time step
    for i in range(N):
        inv_sqrt_F_i = np.linalg.inv(sqrtm(F[i]))
        # for every constraint
        for j in range(nx):
            x_tight[j, i+1] = np.linalg.norm(inv_sqrt_F_i @ X.A[j,:].reshape(-1,1), ord=2) * sqrt_p_tilde
        for j in range(nu):
            u_tight[j, i+1] = np.linalg.norm(inv_sqrt_F_i @ K.T @ U.A[j,:].reshape(-1,1), ord=2) * sqrt_p_tilde

    # check that the tightened constraints are valid
    for i in range(N):
        if np.any(X.b - x_tight[:,i] < 0) and np.any(U.b - u_tight[:,i] < 0):
            raise Exception('Infinite Step PRS Set is bigger than the state constraints')

    return x_tight, u_tight, F, p_tilde, P, K
