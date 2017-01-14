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
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.check_depth = check_depth
        
    def check_termination(self,solver):
        to_checkwith = solver.it - solver.func.n_dim*self.check_depth #when checking convergence
        if(to_checkwith > 0):
            abs_diff = solver.bestvals[to_checkwith] - solver.bestvals[-1]
            rel_diff = abs((solver.bestvals[to_checkwith] - solver.bestvals[-1])/solver.bestvals[-1])              
            abs_break = (abs_diff < self.abs_tol)
            rel_break = (rel_diff < self.rel_tol)
            
            if(abs_break or rel_break):
                return True
        return False