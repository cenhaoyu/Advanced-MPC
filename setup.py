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

from setuptools import setup, find_packages

setup(
    name="ampyc",
    version="1.0.0",
    author="Mike Zhang, Jerome Sieber, Alexandre Didier, Rahel Rickenbach",
    packages=find_packages(),
    author_email="jsieber@ethz.ch; adidier@ethz.ch; rrahel@ethz.ch",
    description="Python implementation of various advanced MPC algorithms",
)