# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 17:08:10 2017

@author: Ebbe
"""
#Common interface for the different solvers.

#All solvers initiate the 'func', and an empty list of observers
#All solvers are able to append observers
#All solvers have a 'step_alg()' function, which iterates the internal points and values
#All solvers have a 'solve_alg()' function, which "solves" to the best of its ability, and returns the position and variables
class solver_interface(object):
    def __init__(self,termination_strategies):
        #self.func = func
        self.observers = []
        self.termination_strategies = termination_strategies
    def attach(self,observer):
        self.observers.append(observer)
    
    def step(self):
        [i.notify_step_start() for i in self.observers]
        self.step_alg()
        [i.notify_step_end() for i in self.observers]
        
    def solve(self,func,point_start):
        [i.notify_solve_start() for i in self.observers]
        val, var = self.solve_alg(func,point_start)
        [i.notify_solve_end() for i in self.observers]
        return val, var 