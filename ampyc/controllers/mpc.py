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

class MPC(ControllerBase):
    '''Construct and solve nominal MPC Linear Problem'''

    def __init__(self, sys, params):
        super().__init__(sys, params)

    def _init_problem(self, sys, params):
        # define optimization variables
        self.x = cp.Variable((sys.n, params.N+1))
        self.u = cp.Variable((sys.m, params.N))
        self.x_0 = cp.Parameter((sys.n))

        # define the objective
        objective = 0.0
        for i in range(params.N):
            objective += cp.quad_form(self.x[:, i], params.Q) + cp.quad_form(self.u[:, i], params.R)
        # NOTE: terminal cost is trivially zero due to terminal constraint

        # define the constraints
        constraints = [self.x[:, 0] == self.x_0]
        for i in range(params.N):
            constraints += [self.x[:, i+1] == sys.A @ self.x[:, i] + sys.B @ self.u[:, i]]
            constraints += [sys.X.A @ self.x[:, i] <= sys.X.b]
            constraints += [sys.U.A @ self.u[:, i] <= sys.U.b]
        constraints += [self.x[:, -1] == 0.0]

        # define the CVX optimization problem object
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def _define_output_mapping(self):
        return {
            'control': self.u,
            'state': self.x,
        }

