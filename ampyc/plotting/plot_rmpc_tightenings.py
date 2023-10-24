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
import polytope as pc
import matplotlib.pyplot as plt
from matplotlib import gridspec

from .plot_quad_set import plot_quad_set

def plot_rmpc_tightenings(fig_number, num_steps, X_tight, U_tight, X, U, P, delta, params):
    fig = plt.figure(num=fig_number, figsize=(10,6)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.9]) 

    # Plot the state tightening
    ax = plt.subplot(gs[0])
    X_deg = (180/np.pi) * X
    X_deg.plot(ax=ax, fill=False, edgecolor='k', alpha=1, linewidth=2, linestyle='-')

    (180/np.pi * X_tight).plot(ax=ax, fill=False, edgecolor='k', alpha=1, linewidth=1, linestyle='--')

    plot_quad_set(ax=ax, P=P*(np.pi/180)**2, rho=delta**2, label='RPI set')

    ax.set_xlim(X_deg.xlim)
    ax.set_ylim(X_deg.ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('position [deg]')
    ax.set_ylabel('velocity [deg/s]')
    ax.grid(visible=True)

    # Plot input tightening
    if U.vertices is None:
        U.vertices = pc.extreme(U)
    if U_tight.vertices is None:
        U_tight.vertices = pc.extreme(U_tight)

    ax = plt.subplot(gs[1])
    ax.axline((-1, U.vertices.max()), slope=0, color='k', linewidth=2, linestyle='-', label='original constraints')
    ax.axline((-1, U.vertices.min()), slope=0, color='k', linewidth=2, linestyle='-')
    ax.axline((-1, U_tight.vertices.max()), slope=0, color='k', linewidth=1, linestyle='--', label='tightened constraints')
    ax.axline((-1, U_tight.vertices.min()), slope=0, color='k', linewidth=1, linestyle='--')
    ax.set_xlabel('time')
    ax.set_ylabel('input')
    ax.set_xlim([0, num_steps])
    ax.grid(visible=True)

    # Collect the labels and handles from the subplots
    all_handles = []
    all_labels = []
    for ax in fig.get_axes():
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)
    ax.legend(all_handles, all_labels)
    