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
from scipy.linalg import solve_discrete_are, sqrtm
import cvxpy as cp

### some math helper functions ###

def _compute_tube_controller(sys, Q, R, rho, lam):
    '''
    Computes tube & terminal controller and terminal cost
    for robust adaptive MPC.

    The method is from Appendix A in Köhler et al. (2019),
    "Linear robust adaptive model predictive control:
    Computational complexity and conservatism"
    '''

    vertices_omega = sys.omega.vertices
    num_vertices = vertices_omega.shape[0]

    # look up necessary values
    n, m = (sys.n, sys.m)
    A_delta = sys.A_delta
    B = sys.B
    X, U, W = (sys.X, sys.U, sys.W)
    nx, nu, nw = (X.A.shape[0], U.A.shape[0], W.A.shape[0])

    sqrt_Q, sqrt_R = (sqrtm(Q), sqrtm(R))

    # Computation of a rho-contractive polytope
    # Find feedback K and terminal cost P
    E = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))

    # define the objective
    objective = cp.Minimize(-cp.log_det(E))

    # define the constraints
    constraints = []
    for i in range(num_vertices):
        # compute A for this vertex (extreme point) of omega
        A = np.zeros((n, n))
        A[:] = A_delta[0]
        for j in range(1, len(A_delta)):
            A += A_delta[j] * vertices_omega[i, j-1]

        # lyapunov equation
        lyap_bmat = cp.bmat(
            [
                [E,                (A @ E + B @ Y).T, sqrt_Q @ E,       Y.T @ sqrt_R],
                [(A @ E + B @ Y),  E,                 np.zeros((n, n)), np.zeros((n, m))],
                [(sqrt_Q @ E).T,   np.zeros((n, n)),  np.eye(n),        np.zeros((n, m))],
                [(Y.T @ sqrt_R).T, np.zeros((m, n)),  np.zeros((m, n)), np.eye(m)],

            ]
        )
        constraints += [lyap_bmat >> 0]

        # rho-contractivity
        rho_bmat = cp.bmat(
            [
                [rho * E,         (A @ E + B @ Y).T],
                [(A @ E + B @ Y), rho * E],
            ]
        )
        constraints += [rho_bmat >> 0]

        # RPI condition
        for k in range(nw):
            rpi_bmat = cp.bmat(
                [
                    [lam * E,          np.zeros((n, 1)),              (A @ E + B @ Y).T],
                    [np.zeros((1, n)), np.diag([1 - lam]),            W.vertices[k,:].reshape(1,-1)],
                    [(A @ E + B @ Y),  W.vertices[k,:].reshape(-1,1), E]
                ]
            )
            constraints += [rpi_bmat >> 0]

    # constraint satisfaction
    F = np.concatenate([X.A / X.b.reshape(-1,1), np.zeros((nu, n))], axis=0)
    G = np.concatenate([np.zeros((nx, m)), U.A / U.b.reshape(-1,1)], axis=0)
    for i in range(nx + nu):
        constraints += [cp.bmat([[np.diag([1]), (F[i,:].reshape(1,-1) @ E + G[i,:].reshape(1,-1) @ Y)],
                                 [(F[i,:].reshape(1,-1) @ E + G[i,:].reshape(1,-1) @ Y).T, E]]) >> 0]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()

    P = np.linalg.inv(np.array(E.value))
    K = np.array(Y.value) @ P

    return K, P

def LQR(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return P, K
