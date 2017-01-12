# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:35:38 2017

@author: Ebbe
"""
import numpy as np
from solver import solver_interface
#In each iteration:
#Check the points +/- alpha in each direction, tentatively move to that point, and check other directions
#The 'normal' method is to move along one line first as much as possible, then move on to the next dimension
#But its more fun to spend time developing the more advanced algorithms, come back to this later
class naive_line_search(solver_interface):
    
    def __init__(self, abs_tol = 0.0001, rel_tol = 0.0001, max_iter = 1000,start_size = 0.005,alpha_reduc_factor = 0.8):
        super().__init__()
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.max_iter = max_iter
        
        self.id = "NAIVE_LINE_SEARCH"
        #the size of the stepping algorithm, and the reduction param
        self.alpha = start_size
        self.alpha_reduc_factor = alpha_reduc_factor
            
    def step_alg(self):
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
        
    def solve_alg(self, func,point_start):        
        self.func = func
        self.points = point_start
        
        it = 0
        self.values = self.func.evaluate(self.points)        
        bestvals = [self.values]
        
        while(it < self.max_iter):    
            oldval = self.values
            self.step()
            
            bestvals.append(self.values)
            to_checkwith = it - self.func.n_dim*4 #when checking convergence
            if(to_checkwith > 0):
                abs_diff = bestvals[to_checkwith] - bestvals[-1]
                rel_diff = abs((bestvals[to_checkwith] - bestvals[-1])/bestvals[-1])              
                abs_break = (abs_diff < self.abs_tol)
                rel_break = (rel_diff < self.rel_tol)
                
                if(abs_break or rel_break):
                    print("breaking: ")
                    print("current val: " + str(bestvals[-1]))
                    print("prev val:    " + str(bestvals[to_checkwith]))
                    print("abs diff:    " + str(abs_diff))
                    print("rel diff:    " + str(rel_diff))
                    break
            if(oldval <= self.values):
                self.alpha *= self.alpha_reduc_factor   
            it+=1
        return([self.values, self.points])