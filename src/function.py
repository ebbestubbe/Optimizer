'''
Created on 24/10/2016

@author: Ebbe
'''
import numpy as np
class function_interface(object):
    '''
    n_dim: number of dimensions
    bounds: [lower,upper]
    '''    
    
    def evaluate(self,var):
        self.checkbounds(var)
        assert(self.n_dim == len(var))
    
    def checkbounds(self,var):
        for i in range(len(var)):
            assert(var[i] >= self.bounds[0][i] and var[i] <= self.bounds[1][i])
        