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

from abc import ABC

class EstimatorBase(ABC):
    def __init__(self):
        raise NotImplementedError

    def update_estimate(self,data_x,data_y):
        '''
        This method is implemented by some estimators to update the estimated parameter value.
        '''
        raise NotImplementedError

    def predict(self, x):
        '''
        This method is implemented by some estimators to predict a value for a data point.
        '''
        raise NotImplementedError
