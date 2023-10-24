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

def plot_x_state_time(fig_number, x, X, params, label=None, legend_loc='upper right', title=None):
    # check if the figure number is already open
    if plt.fignum_exists(fig_number):
        fig = plt.figure(fig_number)
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
    else:
        fig, (ax1, ax2) = plt.subplots(2,1, num=fig_number, sharex=True)

    num_steps = x.shape[0]

    ax1.plot(np.rad2deg(x[:,0,:]), color=params.color, alpha=params.alpha, linewidth=params.linewidth, label=label)
    ax1.axline((-1, np.rad2deg(X.vertices[:,0].max())), slope=0, color='k', linewidth=2)
    ax1.axline((-1, np.rad2deg(X.vertices[:,0].min())), slope=0, color='k', linewidth=2)
    ax1.set_ylabel('position [deg]')
    ax1.set_xlim([0, num_steps])
    ax1.grid(visible=True)
    if title is not None:
        ax1.set_title(title)

    if label is not None:
        # remove duplicate legend entries
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc=legend_loc)
    
    ax2.plot(np.rad2deg(x[:,1,:]), color=params.color, alpha=params.alpha, linewidth=params.linewidth)
    ax2.axline((-1, np.rad2deg(X.vertices[:,1].max())), slope=0, color='k', linewidth=2)
    ax2.axline((-1, np.rad2deg(X.vertices[:,1].min())), slope=0, color='k', linewidth=2)
    ax2.set_xlabel('time')
    ax2.set_ylabel('velocity [deg/s]')
    ax2.set_xlim([0, num_steps])
    ax2.grid(visible=True)

def plot_x_state_state(fig_number, x, X, params, label=None, legend_loc='upper right', title=None):
    # check if the figure number is already open
    if plt.fignum_exists(fig_number):
        fig = plt.figure(fig_number)
        ax = fig.axes[0]
    else:
        fig = plt.figure(fig_number)
        ax = plt.gca()

    ax.scatter(np.rad2deg(x[0,0,:]), np.rad2deg(x[0,1,:]), marker='o', facecolors='none', color='k', label='initial state')
    ax.plot(np.rad2deg(x[:,0,:]), np.rad2deg(x[:,1,:]), color=params.color, alpha=params.alpha, linewidth=params.linewidth, label=label)

    X_deg = 180/np.pi * X
    X_deg.plot(ax=ax, fill=False, edgecolor="k", alpha=1, linewidth=2, linestyle='-') 
    ax.set_xlabel('position [deg]')
    ax.set_ylabel('velocity [deg/s]')
    ax.grid(visible=True)
    ax.set_xlim(X_deg.xlim)
    ax.set_ylim(X_deg.ylim)
    if title is not None:
        ax.set_title(title)

    # remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc=legend_loc)