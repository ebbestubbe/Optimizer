# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 21:38:26 2016

@author: Ebbe
"""
import numpy as np
import matplotlib.pyplot as plt

from Solvers.simplex import simplex
from Solvers.naive_line_search import naive_line_search
import Solvers.observers
import Functions.test_functions
from testing_methods import fullreport

def main():
    plt.close("all")
    
    rel_tol = 10e-10
    abs_tol = 10e-10
    rel_tol = 0
    abs_tol = 0
    
    max_iter = 200
    start_size = 0.005
    
    solver1 = simplex(rel_tol = rel_tol, abs_tol = abs_tol, max_iter = max_iter,start_size = start_size)
    
    #fullreport(solver1)
    
    plt.close("all")
    max_iter = 1000
    
    start_size = 0.001
    alpha_reduc_factor = 1.0
    solver2 = naive_line_search(rel_tol = rel_tol, abs_tol = abs_tol, max_iter = max_iter,start_size = start_size,alpha_reduc_factor = alpha_reduc_factor)
    
    fullreport(solver2)
    
if __name__ == '__main__':
    main()
    