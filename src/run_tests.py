# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 21:38:26 2016

@author: Ebbe
"""
import numpy as np
import matplotlib.pyplot as plt

from Solvers.simplex import simplex
from Solvers.naive_line_search import naive_line_search
from Solvers.pattern_search import pattern_search
import Solvers.observers
import Functions.test_functions
from testing_methods import fullreport

from Solvers.terminationstrat import termination_strategy_tolerance
from Solvers.terminationstrat import termination_strategy_max_iter
from Solvers.terminationstrat import termination_strategy_max_eval
def main():
    plt.close("all")
    
    rel_tol = 10e-10
    abs_tol = 10e-10
    check_depth = 4
    #rel_tol = 0
    #abs_tol = 0
    max_eval = 20
    max_iter = 500
    start_size = 0.005
    t_strat_tol = termination_strategy_tolerance(rel_tol = rel_tol, abs_tol = abs_tol, check_depth = check_depth)
    t_strat_max_iter = termination_strategy_max_iter(max_iter = max_iter)
    t_strat_max_eval = termination_strategy_max_eval(max_eval = max_eval)
    termination_strategies = [t_strat_tol,t_strat_max_iter,t_strat_max_eval]
    solver1 = simplex(start_size = start_size,termination_strategies = termination_strategies)    
    fullreport(solver1)
    return
    alpha_reduc_factor = 0.8
    solver2 = naive_line_search(rel_tol = rel_tol, abs_tol = abs_tol, max_iter = max_iter,start_size = start_size,alpha_reduc_factor = alpha_reduc_factor)    

    fullreport(solver2)
    
    solver3 = pattern_search(rel_tol = rel_tol, abs_tol = abs_tol, max_iter = max_iter,start_size = start_size)    
    fullreport(solver3)
   
    
if __name__ == '__main__':
    main()
    