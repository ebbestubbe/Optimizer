# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:14:49 2017

@author: Ebbe
"""
from solver import solver_interface
import numpy as np

class pattern_search(solver_interface):
    
    def __init__(self,start_size = 0.005,termination_strategies = [],reduc_factor = 0.5):
        super().__init__(termination_strategies)
        
        self.id = "PATTERN_SEARCH"
        #the size of the stepping algorithm, and the reduction param
        
        self.start_size = start_size
        self.reduc_factor = reduc_factor

    def step_alg(self):
        point_candidates = []
        value_candidates = []
        for i in range(self.func.n_dim):
            point_mod = np.zeros(self.func.n_dim)
            point_mod[i] = self.step_size
            
            point_neg = np.clip(self.points - point_mod, self.func.bounds[0], self.func.bounds[1])
            value_neg = self.func.evaluate(point_neg)
            
            point_pos = np.clip(self.points + point_mod, self.func.bounds[0], self.func.bounds[1])
            value_pos = self.func.evaluate(point_pos)
            
            point_candidates.append([point_neg, point_pos])
            value_candidates.append([value_neg, value_pos])
        #min in each direction:
        #for each dimension, 0 for negative, 1 for positive
        min_direction = [x.index(min(x)) for x in value_candidates] 
        #The minimum value in each dimension
        min_values = [value_candidates[i][min_direction[i]] for i in range(self.func.n_dim)]
        #The dimension with the lowest func value
        min_dimension = min_values.index(min(min_values))
        #The coordinates to switch to:
        min_coords = [min_dimension,min_direction[min_dimension]]
        
        val_lowest = value_candidates[min_coords[0]][min_coords[1]]
        if(val_lowest < self.values):
            self.values = val_lowest
            self.points = point_candidates[min_coords[0]][min_coords[1]]
        
    def solve_alg(self, func, point_start):
        self.func = func
        self.points = point_start
        self.values = self.func.evaluate(self.points)
        self.step_size = self.start_size
        
        self.it = 0
        self.bestvals = [self.values]
        while(True):
            self.step()
            self.bestvals.append(self.values)
            
            #If there was no improvement: reduce the stepping size:
            if(self.bestvals[-2] <= self.values):
                self.step_size*= self.reduc_factor
            
            break_bools = [i.check_termination(solver=self) for i in self.termination_strategies]
            if(any(break_bools)):
                break
            
            self.it+=1   
            
        return([self.values,self.points])