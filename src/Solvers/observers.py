# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 19:09:01 2016

@author: Ebbe
"""
import numpy as np
import timeit

class observer(object):
    def notify_solve_start(self):
        pass
    def notify_solve_end(self):
        pass
    def notify_reflecting(self):
        pass    
    def notify_expanding(self):
        pass
    def notify_contracting(self):
        pass
    def notify_shrinking(self):
        pass
    def notify_step_start(self):        
        pass
    def notify_step_end(self):
        pass
    def get_result(self):
        pass
    
class observer_simplex_print_log(observer):        
    def notify_solve_start(self):
        print("Started solving")
    def notify_solve_end(self):
        print("Ending solving")
    def notify_reflecting(self):
        print("Reflecting")    
    def notify_expanding(self):
        print("Expanding")
    def notify_contracting(self):
        print("Contracting")
    def notify_shrinking(self):
        print("Shrinking")        

class observer_timeit(observer):    
    def __init__(self):
        #The variables to be returned
        self.steptimes = []
        self.solvetime = None
        
        #Tentative starttimes used to calculate the endtimes.
        self.step_starttime = None
        self.solve_starttime = None
        
    def notify_solve_start(self):
        self.solve_starttime = timeit.default_timer()
    def notify_solve_end(self):
        solve_endtime = timeit.default_timer()
        self.solvetime = solve_endtime - self.solve_starttime
    def notify_step_start(self):
        self.step_starttime = timeit.default_timer()
    def notify_step_end(self):
        step_endtime = timeit.default_timer()
        self.steptimes.append(step_endtime - self.step_starttime)
    def get_steptimes(self):
        return self.steptimes
    def get_solvetime(self):
        return self.solvetime
        
#save the minimum function value at each step:
class observer_step_log(observer):
    def __init__(self,solver):
        self.result = []
        self.solver = solver
    def notify_step_end(self):
        n_eval = self.solver.func.n_evaluations
        points = self.solver.bestpoint
        vals = self.solver.bestvalue
        self.result.append([vals,points,n_eval])
    def get_result(self):
        return self.result

#prints a log of the time progress, iterations, evaluations, and estimations for the ending time, if there is such a thing.
#REMAKE: take the evaluation and iteration number from termination strategies directly
class observer_step_print(observer):
    def __init__(self,solver,n_step_interval = 1):
        self.n_step_interval = n_step_interval
        self.solver = solver
        
        self.max_it = None
        self.max_eval = None
    
    def notify_solve_start(self):
        self.time_starttime = timeit.default_timer()
        t_strat = self.solver.termination_strategies
        for i in range(len(t_strat)):
            if(t_strat[i].id == "TERMINATION_STRATEGY_MAX_ITER"):
                self.max_it = t_strat[i].max_iter
            if(t_strat[i].id == "TERMINATION_STRATEGY_MAX_EVAL"):
                self.max_eval = t_strat[i].max_eval
                        
                
    def notify_step_end(self):
        print_bool = (self.solver.it % self.n_step_interval) == 0 #Check to see if we have to print
        
        if(print_bool):
            step_time = timeit.default_timer()
            time_elapsed = step_time - self.time_starttime
            outputstring = "time: " + str(round(time_elapsed/60)) + " min; it: " + str(self.solver.it) + "; eval: " + str(self.solver.func.n_evaluations) + ";"
            
            estimations = []
            if (self.max_it != None):
                time_per_it = time_elapsed/(self.solver.it+1)
                est_it = (self.max_it - self.solver.it)*time_per_it
                estimations.append(est_it)
            if (self.max_eval != None):
                time_per_eval = time_elapsed/self.solver.func.n_evaluations
                est_eval = (self.max_eval - self.solver.func.n_evaluations)*time_per_eval
                estimations.append(est_eval)
            if(self.max_it != None or self.max_eval != None):
                outputstring += " time left: " + str(round(min(estimations)/60)) + " min;"
            print(outputstring)
            
#The results are given in a the data format [step iteration][points//vals][simplex corner]
#so [results[i][1][0] for i in range(len(results))] gives the best value in each iteration        
class observer_population_log(observer):    
    def __init__(self,solver):    
        self.result = []
        self.solver = solver
    def notify_step_end(self):
        pop = self.solver.population
        vals = self.solver.population_values
        self.result.append([pop,vals])
    def get_result(self):
        return self.result
