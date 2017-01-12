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

def main():
    '''
    c = [[0,-8],[-2,6],[3,-9]]
    
    b = [x.index(min(x)) for x in c]
    d = [c[i][b[i]] for i in range(len(b))]
    print(b)
    print(d)
    minval_ind = d.index(min(d))
    print(minval_ind)
    minval_coord = [minval_ind, b[minval_ind]]
    print(minval_coord)
    print(c[minval_coord[0]][minval_coord[1]])
    return
    '''
    plt.close("all")
    
    rel_tol = 10e-10
    abs_tol = 10e-10
    rel_tol = 0
    abs_tol = 0
    
    max_iter = 2000
    start_size = 0.5
    '''
    solver1 = simplex(rel_tol = rel_tol, abs_tol = abs_tol, max_iter = max_iter,start_size = start_size)
    
    #fullreport(solver1)
    
    plt.close("all")
    max_iter = 1000
    
    start_size = 0.001
    alpha_reduc_factor = 1.0
    solver2 = naive_line_search(rel_tol = rel_tol, abs_tol = abs_tol, max_iter = max_iter,start_size = start_size,alpha_reduc_factor = alpha_reduc_factor)
    
    fullreport(solver2)
    '''
    solver3 = pattern_search(rel_tol = rel_tol, abs_tol = abs_tol, max_iter = max_iter,start_size = start_size)
    
    fullreport(solver3)
   
    
if __name__ == '__main__':
    main()
    