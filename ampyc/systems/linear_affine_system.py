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

from .system_base import SystemBase
import numpy as np

from ampyc.utils import Polytope

class LinearAffineSystem(SystemBase):

    def __init__(self, params):
        self.update_params(params)

    def update_params(self, params):
        super().update_params(params)

        # NOTE: This class only stores the initial estimate of theta and omega given by the params
        self.update_omega(Polytope(params.A_theta, params.b_theta))
        self.theta = params.theta
        self.num_uncertain_params = params.num_uncertain_params

        for A, B in zip(params.A_delta, params.B_delta):
            assert A.shape == (self.n, self.n), 'component of A_delta must have shape (n,n)'
            assert B.shape == (self.n, self.m), 'component of B_delta must have shape (n,m)'
        self.A_delta = params.A_delta
        self.B_delta = params.B_delta

        # NOTE: This class only stores A and B based on the initial estimate of theta from the params
        # they're not updated
        self.A = np.zeros((self.n, self.n))
        self.A[:] = self.A_delta[0]
        for i in range(1, len(self.A_delta)):
            self.A += self.A_delta[i] * self.theta[i-1]
        
        self.B = np.zeros((self.n, self.m))
        self.B[:] = self.B_delta[0]
        for i in range(1, len(self.B_delta)):
            self.B += self.B_delta[i] * self.theta[i-1]
        
        assert params.C.shape[1] == self.n, 'C must have shape (num_output, n)'
        assert params.D.shape[1] == self.m, 'D must have shape (num_output, m)'
        self.C = params.C
        self.D = params.D

    def update_omega(self, omega):
        self.omega = omega
        self.omega.Vrep()

    def f(self, x, u):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        return self.A @ x.reshape(self.n, 1) + self.B @ u.reshape(self.m, 1)

    def h(self, x, u):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        return self.C @ x.reshape(self.n, 1) + self.D @ u.reshape(self.m, 1)
        