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

def plot_u(fig_number, u, U, params, label=None):
    num_steps = u.shape[0]
    fig = plt.figure(fig_number)
    plt.plot(u, color=params.color, alpha=params.alpha, linewidth=params.linewidth, label=label)
    plt.axline((-1, U.vertices.max()), slope=0, color='k', linewidth=2)
    plt.axline((-1, U.vertices.min()), slope=0, color='k', linewidth=2)
    plt.xlabel('time')
    plt.ylabel('input')
    plt.xlim([0, num_steps])
    plt.grid(visible=True)