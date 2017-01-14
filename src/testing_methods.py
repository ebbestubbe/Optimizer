# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 23:07:48 2017

@author: Ebbe
"""
import numpy as np
import matplotlib.pyplot as plt

from Solvers.simplex import simplex
import Solvers.observers
import Functions.test_functions

#Use this to run on all test functions, and report everything necessary
def fullreport(solver): 
    optfuncs = []
    
    optfuncs.append(test_sphere())
    '''
    optfuncs.append(test_rosenbrock())
    optfuncs.append(test_himmelblau())
    optfuncs.append(test_rastrigin())
    optfuncs.append(test_bukin6())
    optfuncs.append(test_eggholder())
    optfuncs.append(test_cross_in_tray())
    '''
    for i in range(len(optfuncs)):
        report(optfuncs[i][0],optfuncs[i][1],solver)
    return

def test_sphere():
    optfunc = Functions.test_functions.sphere(2)
    startpoint = np.array([2,8])
    return [optfunc,startpoint]
    
def test_rosenbrock():
    optfunc = Functions.test_functions.rosenbrock()
    startpoint = np.array([-1,1])
    return [optfunc,startpoint]
    
def test_himmelblau():
    optfunc = Functions.test_functions.himmelblau()
    startpoint = np.array([-1,1])
    return [optfunc,startpoint]
    
def test_rastrigin():
    optfunc = Functions.test_functions.rastrigin(2)
    startpoint = np.array([0.4,-0.3])   
    return [optfunc,startpoint]
    
def test_bukin6():
    optfunc = Functions.test_functions.bukin6()
    startpoint = np.array([-12,2])   
    return [optfunc,startpoint]
    
def test_eggholder():
    optfunc = Functions.test_functions.eggholder()
    startpoint = np.array([511,400])   
    return [optfunc,startpoint]
    
def test_cross_in_tray():
    optfunc = Functions.test_functions.cross_in_tray()
    startpoint = np.array([1.2,1.2])   
    return [optfunc,startpoint]
    
def report(optfunc,startpoint,solver):
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
    if(solver.id == "NELDER_MEAD_SIMPLEX"):
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
    
    #optfunc.contour(optfunc.bounds[0],optfunc.bounds[1],points = 100,N=15)
    
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
    
    #optfunc.contour([lower_x, lower_y],[upper_x, upper_y],points = 100,N=15)
    
    plt.show()
    
    '''    
    plt.figure(4)
    plt.subplot(2,1,1)
    plt.plot(steptimes)
    
    plt.subplot(2,1,2)
    plt.hist(steptimes)
    plt.show()
    ''' 
    print("val found :" + str(val))
    print("optimum   :" + str(optfunc.min_vals[0]))
    print("difference:" + str(abs(val-optfunc.min_vals[0])))
    print("var:")
    print(var)
    print("solvetime: " + str(solvetime))
    print("avg steptimes: " + str(np.mean(steptimes)))