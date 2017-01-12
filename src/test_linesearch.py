# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 21:38:26 2016

@author: Ebbe
"""
import numpy as np
import matplotlib.pyplot as plt

from Solvers.naive_line_search import naive_line_search
import Solvers.observers
import Functions.test_functions


def main():
    plt.close("all")
    
    test_sphere()
    test_rosenbrock()
    test_himmelblau()
    test_rastrigin()
    test_bukin6()
    test_eggholder()
    test_cross_in_tray()

def test_sphere():
    optfunc = Functions.test_functions.sphere(2)
    startpoint = np.array([-2.31,8.2531])
    test_func(optfunc,startpoint)

def test_rosenbrock():
    optfunc = Functions.test_functions.rosenbrock()
    startpoint = np.array([-1,1])
    test_func(optfunc,startpoint)

def test_himmelblau():
    optfunc = Functions.test_functions.himmelblau()
    startpoint = np.array([-1,1])
    test_func(optfunc,startpoint)
    
def test_rastrigin():
    optfunc = Functions.test_functions.rastrigin(2)
    startpoint = np.array([0.4,-0.3])   
    test_func(optfunc,startpoint)

def test_bukin6():
    optfunc = Functions.test_functions.bukin6()
    startpoint = np.array([-12,2])   
    test_func(optfunc,startpoint)
    
def test_eggholder():
    optfunc = Functions.test_functions.eggholder()
    startpoint = np.array([511,400])   
    test_func(optfunc,startpoint)

def test_cross_in_tray():
    optfunc = Functions.test_functions.cross_in_tray()
    startpoint = np.array([1.158,1.2])   
    test_func(optfunc,startpoint)

def test_func(optfunc,startpoint):
    #optfunc =  Functions.test_functions.rosenbrock()
    #startpoint = np.arr-ay([-1,1])
    rel_tol = 10e-10
    abs_tol = 10e-10
    max_iter = 1000
    start_size = 0.1
    alpha_reduc_factor = 0.8
    solver = naive_line_search(optfunc,startpoint,rel_tol = rel_tol, abs_tol = abs_tol, max_iter = max_iter,start_size = start_size,alpha_reduc_factor = alpha_reduc_factor)
    report(optfunc,solver)

def report(optfunc,solver):
    #print("Testing function:")
    #print(optfunc.report())
    
    observer_step_log = Solvers.observers.observer_simplex_step_log(solver)
    observer_time = Solvers.observers.observer_timeit()
    #observer_log = Solvers.observers.observer_simplex_print_log()
    
    solver.attach(observer_step_log)
    solver.attach(observer_time)
    #solver.attach(observer_log)
    
    val,var = solver.solve()
    
    solvetime = observer_time.get_solvetime()
    steptimes = observer_time.get_steptimes()
    
    results = observer_step_log.get_result()
    
    plt.figure(1)
    plt.plot([results[i][1] for i in range(len(results))])
    plt.show()
    
    
    plt.figure(2)
    for i in range(len(results)):
        plt.plot(results[i][0][0],results[i][0][1],'b.')
    plt.show()
    
    plt.figure(3)
    plt.subplot(2,1,1)
    plt.plot(steptimes)
    
    plt.subplot(2,1,2)
    plt.hist(steptimes)
    plt.show()
    
    print("val:")
    print(val)
    print("var:")
    print(var)
    print("solvetime: " + str(solvetime))
    print("avg steptimes: " + str(np.mean(steptimes)))
    
if __name__ == '__main__':
    main()
    