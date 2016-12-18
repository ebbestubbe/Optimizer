# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 19:09:01 2016

@author: Ebbe
"""
import numpy as np

class observer_simplex_print_log():        
    def notify_solve(self):
        print("Started solving")
    def notify_reflecting(self):
        print("Reflecting")    
    def notify_expanding(self):
        print("Expanding")
    def notify_contracting(self):
        print("Contracting")
    def notify_shrinking(self):
        print("Shrinking")
        
    def notify_step(self):
        pass

class observer_simplex_step_log():
    
    def __init__(self,simplex):    
        self.result = []
        self.simplex = simplex
    def notify_solve(self):
        pass
    def notify_reflecting(self):
        pass    
    def notify_expanding(self):
        pass
    def notify_contracting(self):
        pass
    def notify_shrinking(self):
        pass
    def notify_step(self):
        points = self.simplex.points
        vals = np.array([self.simplex.func.evaluate(points[i]) for i in range(len(points))])
        self.result.append([points,vals])
    def get_result(self):
        return self.result