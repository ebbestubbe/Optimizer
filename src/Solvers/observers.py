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

#The results are given in a the data format [step iteration][points//vals][simplex corner]
#so [results[i][1][0] for i in range(len(results))] gives the best value in each iteration
        
class observer_simplex_step_log(observer):    
    def __init__(self,optimizer):    
        self.result = []
        self.optimizer = optimizer
    def notify_step_end(self):
        points = self.optimizer.points
        vals = self.optimizer.values
        self.result.append([points,vals])
    def get_result(self):
        return self.result

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