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
from Solvers.genetic_algorithm import genetic_algorithm

import Solvers.observers
import Functions.test_functions
from testing_methods import fullreport

import testing_methods

from Solvers.terminationstrat import termination_strategy_tolerance
from Solvers.terminationstrat import termination_strategy_max_iter
from Solvers.terminationstrat import termination_strategy_max_eval
def main():
    '''
    mask = np.array([0, 1, 0, 1, 0, 0])
    mask = mask.astype(bool)
    #mask = np.array([False, True, False, True, False, False])
    print(mask)
    v1 = np.array([0.1, 1.2, 2.3,3.4,4.5,5.8])
    v2 = np.array([-0.3,-1.6, -2.6, -3.7, -4.02, -5.1])
    
    v1[~mask] = 0
    print("v1 part: " + str(v1))        
    #v2_copy = np.copy(v2)
    #print("v2 copy: " + str(v2_copy))

    v2[mask] = 0
    print("v2 part: " + str(v2))
    v_child =  v1 +  v2    
    return
    '''
    rel_tol = 10e-10
    abs_tol = 10e-10
    
    check_depth = 30
    #rel_tol = 0
    #abs_tol = 0
    max_eval = 10000000
    max_iter = 5000
    t_strat_tol = termination_strategy_tolerance(rel_tol = rel_tol, abs_tol = abs_tol, check_depth = check_depth)
    t_strat_max_iter = termination_strategy_max_iter(max_iter = max_iter)
    t_strat_max_eval = termination_strategy_max_eval(max_eval = max_eval)
    termination_strategies = [t_strat_tol,t_strat_max_iter,t_strat_max_eval]
    
    pop_size = 100
    ga = genetic_algorithm(pop_size = pop_size,termination_strategies = termination_strategies)    
    fullreport(ga)

def comp():
    solvers = testing_methods.makeallsolvers()
    optfunc = Functions.test_functions.rosenbrock()
    start_point = np.array([-1,1])
    testing_methods.comparesolvers(solvers,optfunc,start_point)

def testall():
    plt.close("all")
    
    rel_tol = 10e-10
    abs_tol = 10e-10
    
    check_depth = 3
    #rel_tol = 0
    #abs_tol = 0
    max_eval = 5000
    max_iter = 3000
    start_size = 0.05
    t_strat_tol = termination_strategy_tolerance(rel_tol = rel_tol, abs_tol = abs_tol, check_depth = check_depth)
    t_strat_max_iter = termination_strategy_max_iter(max_iter = max_iter)
    t_strat_max_eval = termination_strategy_max_eval(max_eval = max_eval)
    termination_strategies = [t_strat_tol,t_strat_max_iter,t_strat_max_eval]
    
    solver1 = simplex(start_size = start_size,termination_strategies = termination_strategies)    
    fullreport(solver1)
    reduc_factor = 0.8
    
    solver2 = naive_line_search(start_size = start_size, termination_strategies = termination_strategies,reduc_factor = reduc_factor)    
    fullreport(solver2)
 
    solver3 = pattern_search(start_size = start_size,termination_strategies = termination_strategies,reduc_factor = reduc_factor)    
    fullreport(solver3)
    
if __name__ == '__main__':
    main()