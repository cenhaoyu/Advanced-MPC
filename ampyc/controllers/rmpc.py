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

from .controller_base import ControllerBase
import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm

class RMPC(ControllerBase):
    '''Construct and solve robust linear MPC Problem'''

    def __init__(self, sys, params, *args, **kwargs):
        super().__init__(sys, params, *args, **kwargs)

    def _init_problem(self, sys, params, rho=0.9):
        # compute tightening
        x_tight, u_tight, P, self.K, delta = self.compute_tightening(rho)
        x_tight = x_tight.flatten()
        u_tight = u_tight.flatten()

        # define optimization variables
        self.z = cp.Variable((sys.n, params.N+1))
        self.v = cp.Variable((sys.m, params.N))
        self.x_0 = cp.Parameter((sys.n))

        # define the objective
        objective = 0.0
        for i in range(params.N):
            objective += cp.quad_form(self.z[:, i], params.Q) + cp.quad_form(self.v[:, i], params.R)
        # NOTE: terminal cost is trivially zero due to terminal constraint

        # define the constraints
        constraints = [cp.norm(sqrtm(P) @ (self.x_0 - self.z[:, 0])) <= delta]
        for i in range(params.N):
            constraints += [self.z[:, i+1] == sys.A @ self.z[:, i] + sys.B @ self.v[:, i]]
            constraints += [sys.X.A @ self.z[:, i] <= sys.X.b - x_tight]
            constraints += [sys.U.A @ self.v[:, i] <= sys.U.b - u_tight]
        constraints += [self.z[:, -1] == 0.0]

        # define the CVX optimization problem object
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def compute_tightening(self, rho):
        ''' 
            Computes an RPI set and the corresponding tightening, 
            which minimizes the constraint tightening.
        '''
        # system dimensions
        n = self.sys.n
        m = self.sys.m

        # system matrices
        A = self.sys.A
        B = self.sys.B

        # state, input, and disturbance sets
        X = self.sys.X
        U = self.sys.U
        W = self.sys.W
        nx = X.A.shape[0]
        nu = U.A.shape[0]

        # setup and solve the offline optimization problem
        E = cp.Variable((n, n),symmetric=True)
        Y = cp.Variable((m, n))

        c_x_2 = cp.Variable((nx, 1))
        c_u_2 = cp.Variable((nu, 1))
        bar_w_2 = cp.Variable()

        # define constraints
        constraints = []
        constraints += [E >> np.diag(np.ones(n))]
        
        E_bmat = cp.bmat(
            [
                [rho**2 * E,   (A @ E + B @ Y).T],
                [(A @ E + B @ Y), E]
            ]
        )
        constraints += [E_bmat >> 0]

        for i in range(nx):
            x_bmat = cp.bmat(
                [
                    [cp.reshape(c_x_2[i],(1,1)),      X.A[i, :].reshape(1,-1) @ E],
                    [E.T @ X.A[i, :].reshape(1,-1).T, E]
                ]
            )
            constraints += [x_bmat >> 0]

        for i in range(nu):
            u_bmat = cp.bmat(
                [
                    [cp.reshape(c_u_2[i],(1,1)),      U.A[i, :].reshape(1,-1) @ Y],
                    [Y.T @ U.A[i, :].reshape(1,-1).T, E]

                ]
            )
            constraints += [u_bmat >> 0]

        for i in range(W.vertices.shape[0]):
            w_bmat = cp.bmat(
                [
                    [cp.reshape(bar_w_2,(1,1)),        W.vertices[i, :].reshape(1,-1)],
                    [W.vertices[i, :].reshape(1,-1).T, E]
                ]
            )
            constraints += [w_bmat >> 0]

        # define objective
        '''
            Please note that we included here a weighting on the state
            tightening, i.e., 50*sum(c_x_2). We did this since for this
            specific example, the cost favours the input tightening and
            including this weighting puts more emphasis on the state
            tightening, therefore ensuring more balance between the two
            terms.
        '''
        objective = cp.Minimize(
            (50*cp.sum(c_x_2) + cp.sum(c_u_2) + (nx + nu) * bar_w_2) / (2 * (1 - rho))
        )

        # solve the problem
        cp.Problem(objective, constraints).solve(solver='SCS')

        # recover lyapunov function and controller
        P = np.linalg.inv(np.array(E.value))
        K = np.array(Y.value) @ P

        # compute delta
        delta = np.sqrt(bar_w_2.value) / (1 - rho)

        # compute tightening of state constraints
        x_tight = delta * np.sqrt(c_x_2.value)

        # compute tightening of input constraints
        u_tight = delta * np.sqrt(c_u_2.value)

        return x_tight, u_tight, P, K, delta
    
    def _define_output_mapping(self):
        return {
            'control': self.v,
            'state': self.z,
        }
