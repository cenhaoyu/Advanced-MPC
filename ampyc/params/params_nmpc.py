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
import casadi
from ampyc.noise import PolytopeNoise
from ampyc.utils import Polytope

class NonlinearMPCParams:

    #TODO: Define unspecified parameters as laid out in the exercise (i.e. fill out all the values marked with None).
    # Do not change the structure of the classes.

    # ---- start inserting here ----
    class ctrl:
        name = 'nominal non-linear MPC'
        N = 10
        Q = 100 * np.eye(2)
        R = 10 * np.eye(1)

    class sys:
        # system dimensions
        n = 2
        m = 1
        dt = 0.1

        # nonlinear dynamics
        k = 4
        g = 9.81
        l = 1.3
        c = 1.5

        def _segway_f(
                x, u, 
                dt=dt, k=k, g=g, l=l, c=c):
            x_next = casadi.vertcat(
                x[0]+dt*x[1],
                x[1]+dt*(-k*x[0]-c*x[1]+g/l*np.sin(x[0])+u)
            )
            return x_next
        
        f = lambda x, u: NonlinearMPCParams.sys._segway_f(x, u)

        # state constraints
        A_x = np.array(
            [
                [1, 0], 
                [-1, 0],
                [0, 1],
                [0, -1]
            ])
        b_x = np.array([np.deg2rad(45), np.deg2rad(45), np.deg2rad(60), np.deg2rad(60)]).reshape(-1,1)

        # input constraints
        A_u = np.array([1, -1]).reshape(-1,1)
        b_u = np.array([5, 5]).reshape(-1,1)
        
        # ---- stop inserting here ----

        # noise description
        A_w = np.array(
            [
                [1, 0], 
                [-1, 0],
                [0, 1],
                [0, -1]
            ])
        b_w = np.array([1e-6, 1e-6, 1e-6, 1e-6]).reshape(-1,1)

        # noise distribution
        noise_generator = PolytopeNoise(Polytope(A_w, b_w))

    class sim:
        num_steps = 30
        num_traj = 1
        x_0 = np.array([np.deg2rad(20), 0]).reshape(-1,1)

    class plot:
        show = True
        color = 'red'
        alpha = 1.0
        linewidth = 1.0
