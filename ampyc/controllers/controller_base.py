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

from abc import ABC, abstractclassmethod
import cvxpy as cp
import casadi

class ControllerBase(ABC):

    def __init__(self, sys, params, *args, **kwargs):
        self.sys = sys
        self.params = params
        self._init_problem(sys, params, *args, **kwargs)
        self.output_mapping = self._define_output_mapping()

    def _init_problem(self, sys, params, *args, **kwargs):
        '''
        This method must be implemented by the controller to define the optimization problem
        '''
        raise NotImplementedError

    def _set_additional_parameters(self, additional_parameters):
        '''
        Some controllers require setting additional parameters of the optimization problem beside just setting the initial condition

        For controllers which require additional parameters, they must override this method
        to set the value of those parameters

        This method will be called to set the additional parameters right before calling the solver
        '''
        pass

    @abstractclassmethod
    def _define_output_mapping(self):
        '''
        Depending on the controller, the final output of the controller may correspond to 
        different variables.

        In case of MPC controllers, all controllers should output the optimal control sequence and the predicted state
        trajectory over the horizon.

        This method must be implemented by the controller to define the mapping from the optimization
        variables to the outputs.
        '''

        ''' TEMPLATE
        return {
            'control': # planned control input trajectory,
            'state': # planned state trajectory
        }
        '''
        
        raise NotImplementedError

    def solve(self, x, additional_parameters={}, verbose=False, solver=None):
        if self.prob != None:
            if not hasattr(self, 'x_0'):
                raise Exception(
                    'The MPC problem must define the initial condition as an optimization parameter self.x_0')
            
            if isinstance(self.prob,cp.Problem):
                try:
                    self.x_0.value = x
                    self._set_additional_parameters(additional_parameters)
                    self.prob.solve(verbose=verbose, solver=solver)

                    if self.prob.status != cp.OPTIMAL:
                        error_msg = 'Solver did not achieve an optimal solution. Status: {0}'.format(self.prob.status)
                        control, state = (None, None)
                    else:
                        error_msg = None
                        control = self.output_mapping['control'].value
                        state = self.output_mapping['state'].value
                except Exception as e:
                    error_msg = 'Solver encountered an error. {0}'.format(e)
                    control, state = (None, None)

            elif isinstance(self.prob, casadi.Opti):
                if verbose:
                    opts = {'ipopt.print_level': 5, 'print_time': 1}
                else:
                    opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
                self.prob.solver('ipopt', opts)

                # casadi will raise an exception if solve() detects an infeasible problem
                try:
                    self.prob.set_value(self.x_0, x)
                    self._set_additional_parameters(additional_parameters)
                    sol = self.prob.solve()
                    if sol.stats()['success']:
                        error_msg = None
                        control = sol.value(self.output_mapping['control'])
                        state = sol.value(self.output_mapping['state'])

                    else:
                        error_msg = 'Solver was not successful with return status: {0}'.format(sol.stats()['return_status'])
                        control, state = (None, None)
                except Exception as e:
                    error_msg = 'Solver encountered an error. {0}'.format(e)
                    control, state = (None, None)

            else:
                raise Exception('Optimization problem type not supported!')
        else:
            raise Exception('Optimization problem is not initialised!')

        return control, state, error_msg
