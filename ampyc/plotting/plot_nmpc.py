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
import matplotlib.pyplot as plt

def plot_nmpc(fig_number, x, infeasible_traj, X):
    fig = plt.figure(fig_number)
    ax=plt.gca()
    cmap = plt.get_cmap('winter')

    ax.scatter(np.rad2deg(x[0,~infeasible_traj]), np.rad2deg(x[1,~infeasible_traj]), 
               marker='o', c='green', label='feasible initial conditions', zorder=3)

    ax.scatter(np.rad2deg(x[0,infeasible_traj]), np.rad2deg(x[1,infeasible_traj]), 
               marker='o', c='darkorange', label='infeasible initial conditions', zorder=3)

    X_deg = (180/np.pi) * X
    X_deg.plot(ax, fill=False, edgecolor='k', alpha=1, linewidth=2, linestyle='-') 
    plt.xlabel('position [deg]')
    plt.ylabel('velocity [deg/s]')
    ax.set_xlim(X_deg.xlim)
    ax.set_ylim(X_deg.ylim)
    plt.legend(loc='upper right')
    plt.grid(visible=True)