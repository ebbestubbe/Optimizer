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
    
    def __init__(self,start_size = 0.005,termination_strategies = [],reduc_factor = 0.8):
        super().__init__(termination_strategies)
        
        self.id = "NAIVE_LINE_SEARCH"
        #the size of the stepping algorithm, and the reduction param
        self.start_size = start_size
        self.reduc_factor = reduc_factor
            
    def step_alg(self):
        for i in range(self.func.n_dim):
            
            point_mod = np.zeros(self.func.n_dim)
            point_mod[i] = self.step_size
            
            #Check the positive direction. If the value is better, go in that direction
            #If the positive direction is not better, check the negative direction
            point_pos = np.clip(self.bestpoint + point_mod, self.func.bounds[0], self.func.bounds[1])
            value_pos = self.func.evaluate(point_pos)
            
            if(value_pos <= self.bestvalue):
                self.bestpoint = point_pos
                self.bestvalue = value_pos
                continue
            point_neg = np.clip(self.bestpoint - point_mod, self.func.bounds[0], self.func.bounds[1])
            value_neg = self.func.evaluate(point_neg)
            
            if(value_neg <= self.bestvalue):
                self.bestpoint = point_neg
                self.bestvalue = value_neg
                continue
        
    def solve_alg(self, func,point_start):        
        self.func = func
        self.bestpoint = point_start
        self.step_size = self.start_size
        self.it = 0
        self.bestvalue = self.func.evaluate(self.bestpoint)        
        self.bestvals = [self.bestvalue]
        
        while(True):    
            self.step()
            self.bestvals.append(self.bestvalue)
            #If there was no improvement, reduce the step size
            if(self.bestvals[-2] <= self.bestvalue):
                self.step_size *= self.reduc_factor   
            
            break_bools = [i.check_termination(solver=self) for i in self.termination_strategies]
            if(any(break_bools)):
                break
            self.it+=1
        return([self.bestvalue, self.bestpoint])