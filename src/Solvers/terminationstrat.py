# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:05:48 2017

@author: Ebbe
"""

#Terminates if the improvement is slow.
#if the relative tolerance or the absolute tolerance goes below the assigned values, returns false.
#check_depth is how many steps back the algorithm checks to see the convergence.
#It is too unstable to check this every step
class termination_strategy_tolerance(object):
    def __init__(self,rel_tol = 10e-5, abs_tol = 10e-5, check_depth = 4):
        self.id = "TERMINATION_STRATEGY_TOLERANCE"
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.check_depth = check_depth
        
    def check_termination(self,solver):
        to_checkwith = solver.it - solver.func.n_dim*self.check_depth #when checking convergence
        if(to_checkwith > 0):
            abs_diff = solver.bestvalues[to_checkwith] - solver.bestvalues[-1]
            rel_diff = abs((solver.bestvalues[to_checkwith] - solver.bestvalues[-1])/solver.bestvalues[-1])              
            #print(abs_diff)
            #print(rel_diff)
            abs_break = (abs_diff < self.abs_tol)
            rel_break = (rel_diff < self.rel_tol)
            
            if(abs_break or rel_break):
                return True
        return False

class termination_strategy_max_iter(object):
    def __init__(self,max_iter):
        self.id = "TERMINATION_STRATEGY_MAX_ITER"
        self.max_iter = max_iter
    
    def check_termination(self,solver):
        if(solver.it >= self.max_iter):
            return True
        return False
        
class termination_strategy_max_eval(object):
    def __init__(self,max_eval):
        self.id = "TERMINATION_STRATEGY_MAX_EVAL"
        self.max_eval = max_eval
        
    def check_termination(self,solver):
        if(solver.func.n_evaluations >= self.max_eval):
            return True
        return False