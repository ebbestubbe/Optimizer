'''
Created on 24/10/2016

@author: Ebbe
'''
import numpy as np
class Random_Solver():
    def __init__(self, func, n_runs):
        self.func = func
        self.n_runs = n_runs
    
    def solve(self):
        lowest_val = np.inf
        lowest_var = [-np.inf]*self.func.n_dim
        
        temp_val = []
        temp_var = []
        
        for i in range(self.n_runs):
            func_var = np.random.rand(self.func.n_dim) #generate n_dim random numbers
            func_val = self.func.evaluate(func_var)            
            if(func_val < lowest_val):
                lowest_var = func_var
                lowest_val = func_val
                
                temp_val.append(lowest_val)
                temp_var.append(lowest_var)
        
        out_val = np.array(temp_val)
        out_var = np.array(temp_var)
        
        return [out_val, out_var]