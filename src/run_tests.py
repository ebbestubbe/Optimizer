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
from testing_methods import fullreport_all

import testing_methods

from Solvers.terminationstrat import termination_strategy_tolerance
from Solvers.terminationstrat import termination_strategy_max_iter
from Solvers.terminationstrat import termination_strategy_max_eval
def main():
    #testall()
    comp()
    return
def comp():
    solvers = testing_methods.makeallsolvers()
    optfunc = Functions.test_functions.himmelblau()
    start_point = np.array([-1,1])
    testing_methods.comparesolvers(solvers,optfunc,start_point)

def testall():
    plt.close("all")
    
    rel_tol = 10e-10
    abs_tol = 10e-10
    
    check_depth = 5
    #rel_tol = 0
    #abs_tol = 0
    max_eval = 3000
    max_iter = 100
    start_size = 0.05
    t_strat_tol = termination_strategy_tolerance(rel_tol = rel_tol, abs_tol = abs_tol, check_depth = check_depth)
    t_strat_max_iter = termination_strategy_max_iter(max_iter = max_iter)
    t_strat_max_eval = termination_strategy_max_eval(max_eval = max_eval)
    termination_strategies = [t_strat_tol,t_strat_max_iter,t_strat_max_eval]
    
    solver1 = simplex(start_size = start_size,termination_strategies = termination_strategies)    
    #fullreport_all(solver1)
    
    reduc_factor = 0.8
    solver2 = naive_line_search(start_size = start_size, termination_strategies = termination_strategies,reduc_factor = reduc_factor)    
    #fullreport_all(solver2)

    solver3 = pattern_search(start_size = start_size,termination_strategies = termination_strategies,reduc_factor = reduc_factor)    
    #fullreport_all(solver3)
    
    pop_size = 100
    solver4 = genetic_algorithm(pop_size = pop_size, termination_strategies = termination_strategies)
    fullreport_all(solver4)
    
if __name__ == '__main__':
    main()