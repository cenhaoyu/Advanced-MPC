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
from ampyc.noise import ZeroNoise

class MPCParams:
    
    class ctrl:
        name = 'nominal MPC'
        N = 10
        Q = 100 * np.eye(2)
        R = 10 * np.eye(1)

    class sys:
        # system dimensions
        n = 2
        m = 1
        dt = 0.1

        # dynamics matrices
        k = 0
        g = 9.81
        l = 1.3 
        c = 0.5
        A = np.array(
            [
                [1, dt],
                [dt*(-k + (g/l)), 1 - dt*c]
            ])
        B = np.array([0, dt]).reshape(-1,1)
        C = np.eye(n)
        D = np.zeros((n,m))

        # state constraints
        A_x = np.array(
            [
                [1, 0], 
                [-1, 0],
                [0, 1],
                [0, -1]
            ])
        b_x = np.array([np.deg2rad(45), np.deg2rad(45), np.deg2rad(45), np.deg2rad(45)]).reshape(-1,1)

        # input constraints
        A_u = np.array([1, -1]).reshape(-1,1)
        b_u = np.array([5, 5]).reshape(-1,1)

        # noise description
        A_w = None
        b_w = None

        # noise distribution
        noise_generator = ZeroNoise(dim=n)

    class sim:
        num_steps = 30
        num_traj = 1
        x_0 = np.array([np.deg2rad(20), np.deg2rad(10)]).reshape(-1,1)

    class plot:
        show = True
        color = 'red'
        alpha = 1.0
        linewidth = 1.0
