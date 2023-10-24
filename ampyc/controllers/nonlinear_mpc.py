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
import casadi

class NonlinearMPC(ControllerBase):
    '''Construct and solve nominal nonlinear MPC Linear Problem'''

    def __init__(self, sys, params):
        super().__init__(sys, params)

    def _init_problem(self, sys, params):
        

        #TODO: 
        # Define the nonlinear nominal MPC problem below.
        # Use the Opti stack from CasADi to formulate the optimization problem.
        # You have to define self.prob!

        # ---- start inserting here
        # init casadi Opti object which holds the optimization problem
        opti = casadi.Opti()

        # define optimization variables     
        self.x =opti.variable(sys.n,params.N+1)
        self.u =opti.variable(sys.m,params.N)
        self.x_0 =opti.parameter(sys.n)

        # define the objective
        objective=0.0
        for i in range(params.N):
            objective += self.x[:, i].T @ params.Q @ self.x[:, i] + \
                         self.u[:, i].T @ params.R @ self.u[:, i]
        # NOTE: terminal cost is trivially zero due to terminal constraint       

        # define the constraints
        constraints = [self.x[:, 0] == self.x_0]
        for i in range(params.N):
            constraints += [self.x[:, i+1] == sys.f(self.x[:, i], self.u[:, i])]
            constraints += [sys.X.A @ self.x[:, i] <= sys.X.b]
            constraints += [sys.U.A @ self.u[:, i] <= sys.U.b]
        constraints += [self.x[:, -1] == 0.0]

        opti.minimize(objective)
        opti.subject_to(constraints)

        self.prob = opti
        self.objective = objective

        # ---- stop inserting here ---- #

    def _define_output_mapping(self):
        return {
            'control': self.u,
            'state': self.x,
        }

