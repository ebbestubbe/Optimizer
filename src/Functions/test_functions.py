'''
Created on 24/10/2016

@author: Ebbe
'''
import numpy as np
from function import function_interface

class sphere(function_interface):

    def __init__(self, n_dim):
        self.n_dim = n_dim
        #self.bounds = [[-np.Inf]*self.n_dim, [np.Inf]*self.n_dim]
        self.bounds = [np.array([-10.0]*self.n_dim), np.array([10.0]*self.n_dim)]
        
    def evaluate(self,var):
        function_interface.evaluate(self,var)
        return np.sum(var**2)
        
#Multidimensional generalizations?
class rosenbrock(function_interface):
    def __init__(self):#,n_dim):
        self.n_dim = 2        
        self.bounds = [[-np.Inf]*self.n_dim, [np.Inf]*self.n_dim]
        
        #Himmelblau parameters:
        self.a = 1
        self.b = 100
            
    def evaluate(self,var):
        function_interface.evaluate(self,var)
        return (self.a - var[0])**2 + self.b*(var[1] - var[0]**2)**2


class himmelblau(function_interface):
    def __init__(self):
        self.n_dim = 2
        self.bounds = [[-10]*self.n_dim, [10]*self.n_dim]
    
    def evaluate(self,var):
        function_interface.evaluate(self,var)
        return (var[0]**2 + var[1] - 11)**2 + (var[0] + var[1]**2 - 7)**2