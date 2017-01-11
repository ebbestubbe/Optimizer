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
from testfunc import fullreport

def main():
    plt.close("all")
    
    rel_tol = 10e-10
    abs_tol = 10e-10
    max_iter = 200
    start_size = 0.005
    
    solver1 = simplex(rel_tol = rel_tol, abs_tol = abs_tol, max_iter = max_iter,start_size = start_size)
    
    #fullreport(solver1)
    
    plt.close("all")
    max_iter = 1000
    
    start_size = 0.1
    alpha_reduc_factor = 0.8
    solver2 = naive_line_search(rel_tol = rel_tol, abs_tol = abs_tol, max_iter = max_iter,start_size = start_size,alpha_reduc_factor = alpha_reduc_factor)
    
    fullreport(solver2)

def report(optfunc,solver,startpoint):
    
    observer_step_log = Solvers.observers.observer_simplex_step_log(solver)
    observer_time = Solvers.observers.observer_timeit()
    #observer_log = Solvers.observers.observer_simplex_print_log()
    
    solver.attach(observer_step_log)
    solver.attach(observer_time)
    #solver.attach(observer_log)
    
    val,var = solver.solve(optfunc,startpoint)
    
    solvetime = observer_time.get_solvetime()
    steptimes = observer_time.get_steptimes()
    
    results = observer_step_log.get_result()
    #reduce results for plotting
    red_results = []
    for i in range(len(results)):
        p = results[i][0][0]
        v = results[i][1][0]
        pair = [p,v]
        red_results.append(pair)
    results = red_results
    
    plt.figure(1)
    plt.plot([results[i][1] for i in range(len(results))])
    plt.show()
    
    plt.figure(2)
    for i in range(len(optfunc.min_points)):
        plt.plot(optfunc.min_points[i][0],optfunc.min_points[i][1],'ro')
    
    for i in range(len(results)):
        plt.plot(results[i][0][0],results[i][0][1],'b.')
    
    optfunc.contour(optfunc.bounds[0],optfunc.bounds[1],points = 100,N=15)
    
    plt.figure(3)
    lower_x = min([results[i][0][0] for i in range(len(results))])
    upper_x = max([results[i][0][0] for i in range(len(results))])
    lower_y = min([results[i][0][1] for i in range(len(results))])
    upper_y = max([results[i][0][1] for i in range(len(results))])
    
    for i in range(len(optfunc.min_points)):
        within_x = (lower_x <= optfunc.min_points[i][0] and upper_x >= optfunc.min_points[i][0])
        within_y = (lower_y <= optfunc.min_points[i][1] and upper_y >= optfunc.min_points[i][1])
        if(within_x and within_y):
            plt.plot(optfunc.min_points[i][0],optfunc.min_points[i][1],'ro')
    
    for i in range(len(results)):
        plt.plot(results[i][0][0],results[i][0][1],'b.')
    
    optfunc.contour([lower_x, lower_y],[upper_x, upper_y],points = 100,N=15)
    
    plt.show()
    '''
    plt.figure(4)
    plt.subplot(2,1,1)
    plt.plot(steptimes)
    
    plt.subplot(2,1,2)
    plt.hist(steptimes)
    plt.show()
    '''
       
    
    print("val:")
    print(val)
    print("var:")
    print(var)
    print("solvetime: " + str(solvetime))
    print("avg steptimes: " + str(np.mean(steptimes)))
    
if __name__ == '__main__':
    main()
    