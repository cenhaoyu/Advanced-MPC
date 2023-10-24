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
from matplotlib.patches import Ellipse

def plot_quad_set(ax, rho, P, xy=(0,0), label=None, alpha=0.4, facecolor='b', edgecolor='black', linewidth=1):
    '''
    Plot the 2D quadratic set of the form x'Px <= rho in ax
    '''
    # Compute the eigenvalues and eigenvectors of P
    eigvals, eigvecs = np.linalg.eig(P)

    # Compute the semi-axes of the ellipse
    a = np.sqrt(rho / eigvals[0])
    b = np.sqrt(rho / eigvals[1])

    # Compute the angle of rotation of the ellipse
    theta = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))

    # Create an Ellipse object
    ellipse = Ellipse(xy=xy, width=2*a, height=2*b, angle=theta, alpha=alpha, edgecolor=edgecolor, facecolor=facecolor,lw=2, label=label, linewidth=linewidth)
    ax.add_patch(ellipse)