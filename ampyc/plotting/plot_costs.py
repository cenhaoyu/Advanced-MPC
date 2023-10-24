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

import matplotlib.pyplot as plt

def plot_costs(fig_number, cost, params, label=None):
    fig = plt.figure(fig_number)
    ax = fig.gca()
    ax.plot(cost, color=params.color, alpha=params.alpha, linewidth=params.linewidth, label=label)
    ax.set_xlabel('time')
    ax.set_ylabel('cost')
    ax.set_title("Cost over Time")
    ax.grid(visible=True)