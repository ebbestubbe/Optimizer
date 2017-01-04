# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:35:38 2017

@author: Ebbe
"""
import numpy as np

class naive_line_search(object):
    
    def __init__(self, func,point_start, abs_tol = 0.0001, rel_tol = 0.0001, max_iter = 1000,start_size = 0.005):
        self.func = func
        
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.max_iter = max_iter
        
        #the size of the stepping algorithm
        self.alpha = start_size
        
        self.observers = []

        self.points = point_start
        self.values = self.func.evaluate(self.points)        
        
    def attach(self,observer):
        self.observers.append(observer)
    
    def step(self):
        [i.notify_step_start() for i in self.observers]
        for i in range(self.func.n_dim):
            
            point_mod = np.zeros(self.func.n_dim)
            point_mod[i] = self.alpha
            
            #Check the positive direction. If the value is better, go in that direction
            #If the positive direction is not better, check the negative direction
            point_pos = np.clip(self.points + point_mod, self.func.bounds[0], self.func.bounds[1])
            value_pos = self.func.evaluate(point_pos)
            
            if(value_pos <= self.values):
                self.points = point_pos
                self.values = value_pos
                continue
            point_neg = np.clip(self.points - point_mod, self.func.bounds[0], self.func.bounds[1])
            value_neg = self.func.evaluate(point_neg)
            
            if(value_neg <= self.values):
                self.points = point_neg
                self.values = value_neg
                continue
        [i.notify_step_end() for i in self.observers]
    def solve(self):
        [i.notify_solve_start() for i in self.observers]
        
        for i in range(self.max_iter):
            oldval = self.values
            self.step()
            if(oldval == self.values):
                self.alpha *= 0.5
        [i.notify_solve_end() for i in self.observers]
        return([self.values, self.points])