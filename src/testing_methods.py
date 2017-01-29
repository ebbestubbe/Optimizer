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

from Solvers.simplex import simplex
from Solvers.naive_line_search import naive_line_search
from Solvers.pattern_search import pattern_search
from Solvers.genetic_algorithm import genetic_algorithm


import Solvers.terminationstrat

def comparesolvers(solvers,optfunc,startpoint):
    colors = ['b','k','g','m']
    plots = [None]*len(solvers)
    for j in range(len(solvers)):
        solver = solvers[j]
        #observer_time = Solvers.observers.observer_timeit()
        observer_step_log = Solvers.observers.observer_step_log(solver)
        #observer_log = Solvers.observers.observer_simplex_print_log()
        
        #solver.attach(observer_step_log)
        #solver.attach(observer_time)
        solver.attach(observer_step_log)
        #solver.attach(observer_log)
        
        val,var = solver.solve(optfunc,startpoint)
        
        #solvetime = observer_time.get_solvetime()
        #steptimes = observer_time.get_steptimes()
        
        result_log = observer_step_log.get_result()
    
        plt.figure(1)
        n_eval = [result_log[i][2] for i in range(len(result_log))]
        func_val = [result_log[i][0] for i in range(len(result_log))]
        plots[j], = plt.plot(n_eval,func_val,colors[j],label= solvers[j].id)
        
        plt.title(optfunc.id)
        plt.xlabel('Number of function evaluations')
        plt.ylabel('Function value')
        
        plt.figure(2)
        for i in range(len(optfunc.min_points)):
            plt.plot(optfunc.min_points[i][0],optfunc.min_points[i][1],'ro')
        
        for i in range(len(result_log)):
            plt.plot(result_log[i][1][0],result_log[i][1][1],colors[j] + ".")
        
        optfunc.contour(optfunc.bounds[0],optfunc.bounds[1],points = 100,N=15)
        plt.xlabel('x')
        plt.ylabel('y')
        
        optfunc.reset_n_evaluations()
    plt.figure(1)
    plt.legend(handles = plots)
        
    plt.show()

def makeallsolvers():
    rel_tol = 10e-10
    abs_tol = 10e-10
    
    check_depth = 5
    #rel_tol = 0
    #abs_tol = 0
    max_eval = 200
    max_iter = 400
    start_size = 0.005
    t_strat_tol = Solvers.terminationstrat.termination_strategy_tolerance(rel_tol = rel_tol, abs_tol = abs_tol, check_depth = check_depth)
    t_strat_max_iter = Solvers.terminationstrat.termination_strategy_max_iter(max_iter = max_iter)
    t_strat_max_eval = Solvers.terminationstrat.termination_strategy_max_eval(max_eval = max_eval)
    termination_strategies = [t_strat_tol,t_strat_max_iter,t_strat_max_eval]
    
    simplex_solver = simplex(start_size = start_size,termination_strategies = termination_strategies)    
    reduc_factor = 0.5
    start_size = 0.05
    linesearch_solver = naive_line_search(start_size = start_size, termination_strategies = termination_strategies,reduc_factor = reduc_factor)
 
    patternsearch_solver = pattern_search(start_size = start_size,termination_strategies = termination_strategies,reduc_factor = reduc_factor)    
    
    pop_size = 20
    ga_solver = genetic_algorithm(pop_size = pop_size,termination_strategies = termination_strategies)
    
    solvers = [simplex_solver,linesearch_solver,patternsearch_solver,ga_solver]
    return solvers
        

#Use this to run on all test functions, and report everything necessary
def fullreport_all(solver): 
    optfuncs = []
    
    optfuncs.append(test_sphere())  
    optfuncs.append(test_rosenbrock())
    optfuncs.append(test_himmelblau())
    optfuncs.append(test_rastrigin())
    optfuncs.append(test_bukin6())
    optfuncs.append(test_eggholder())
    optfuncs.append(test_cross_in_tray())
    
    for i in range(len(optfuncs)):
        fullreport(optfuncs[i][0],optfuncs[i][1],solver)
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
    
def fullreport(optfunc,startpoint,solver):
    
    observer_time = Solvers.observers.observer_timeit()
    observer_step_log = Solvers.observers.observer_step_log(solver)
    
    solver.attach(observer_time)
    solver.attach(observer_step_log)
    
    if(hasattr(solver, 'population')):
 #   if(solver.id == 'NELDER_MEAD_SIMPLEX'):
        observer_pop = Solvers.observers.observer_population_log(solver)
        solver.attach(observer_pop)
    
    val,var = solver.solve(optfunc,startpoint)
    
    result_log = observer_step_log.get_result()
    
    #Plotting function value vs number of evaluations
    plt.figure()
    plt.plot([result_log[i][2] for i in range(len(result_log))],[result_log[i][0] for i in range(len(result_log))])
    plt.title(solver.id + " on " + optfunc.id)
    plt.xlabel('Number of function evaluations')
    plt.ylabel('Function value')
    
    plt.figure()
    for i in range(len(optfunc.min_points)):
        plt.plot(optfunc.min_points[i][0],optfunc.min_points[i][1],'ro')
    
    for i in range(len(result_log)):
        plt.plot(result_log[i][1][0],result_log[i][1][1],'k.')
    
    optfunc.contour(optfunc.bounds[0],optfunc.bounds[1],points = 100,N=15)
    plt.title("Best vals: " + solver.id + " on " + optfunc.id)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    #Find the boundaries for the zoom
    lower_x = min([result_log[i][1][0] for i in range(len(result_log))])
    upper_x = max([result_log[i][1][0] for i in range(len(result_log))])
    lower_y = min([result_log[i][1][1] for i in range(len(result_log))])
    upper_y = max([result_log[i][1][1] for i in range(len(result_log))])
    
    #If all points in the zoom are the same, don't bother doing it(would also crash the contour function plotter anyway)
    if(lower_x != upper_x and lower_y != upper_y):
        plt.figure()
        
        #Plot the minima within
        for i in range(len(optfunc.min_points)):
            within_x = (lower_x <= optfunc.min_points[i][0] and upper_x >= optfunc.min_points[i][0])
            within_y = (lower_y <= optfunc.min_points[i][1] and upper_y >= optfunc.min_points[i][1])
            if(within_x and within_y):
                plt.plot(optfunc.min_points[i][0],optfunc.min_points[i][1],'ro')
        
        for i in range(len(result_log)):
            plt.plot(result_log[i][1][0],result_log[i][1][1],'k.')
        
        optfunc.contour([lower_x, lower_y],[upper_x, upper_y],points = 100,N=15)
        plt.title("Best vals: " + solver.id + " on " + optfunc.id)
        plt.xlabel('x')
        plt.ylabel('y')
    
    #Plot start and end populations, for population based optimizers
    #if(solver.id == "NELDER_MEAD_SIMPLEX"):
    if(hasattr(solver, 'population')):
        result_pop = observer_pop.get_result()
        plt.figure()
        for i in range(len(optfunc.min_points)):
            plt.plot(optfunc.min_points[i][0],optfunc.min_points[i][1],'ro')
        x_pop_start = [result_pop[0][0][i][0] for i in range(len(result_pop[0][1]))]
        y_pop_start = [result_pop[0][0][i][1] for i in range(len(result_pop[0][1]))]
        points_pop_start, = plt.plot(x_pop_start,y_pop_start,color = 'blue',linestyle = 'None',marker = '.',label='pop start')
        
        x_pop_end = [result_pop[-1][0][i][0] for i in range(len(result_pop[0][1]))]
        y_pop_end = [result_pop[-1][0][i][1] for i in range(len(result_pop[0][1]))]
        points_pop_end, = plt.plot(x_pop_end,y_pop_end,color = 'black',linestyle = 'None',marker = '.',label='pop end')
        
        #plt.plot([result_pop[0][0][i][0]] for i in range(len(result_pop[0][1])),[result_pop[0][0][i][1]] for i in range[0][1], 'k.')
        #for i in range(len(result_pop[0][1])):
        #    plt.plot(result_pop[0][0][i][0],result_pop[0][0][i][1],'k.') #print first
        #    plt.plot(result_pop[-1][0][i][0],result_pop[-1][0][i][1],'b.') #print last
        #plt.legend(handles = [points_pop_start,points_pop_end],numpoints = 1)
        optfunc.contour(optfunc.bounds[0],optfunc.bounds[1],points = 100,N=15)
        
        
        plt.title("Populations: " + solver.id + " on " + optfunc.id)
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.show()
        print("Pop_0: blue; Pop_end: black")
        
    
    plt.show()
    
    print("Results for " + optfunc.id)
    print("val found :" + str(val))
    print("optimum   :" + str(optfunc.min_vals[0]))
    print("difference:" + str(abs(val-optfunc.min_vals[0])))
    print("var:")
    
    print(var)
    
    solvetime = observer_time.get_solvetime()
    steptimes = observer_time.get_steptimes()
    print("solvetime: " + str(solvetime))
    print("avg steptimes: " + str(np.mean(steptimes)))