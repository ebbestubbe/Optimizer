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

#Rastrigin function: systematic egg-tray
#minimum at f(0,0,0...,0) = 0
class rastrigin(function_interface):
    def __init__(self,n_dim):
        self.n_dim = n_dim
        self.bounds = [[-5.12]*self.n_dim,[5.12]*self.n_dim]
        
        #Rastrigin parameter
        self.A = 10
    
    def evaluate(self,var):
        function_interface.evaluate(self,var)
        f = self.A*self.n_dim
        for i in range(self.n_dim):
            f += var[i]**2 - self.A*np.cos(2*np.pi*var[i])
        return f

#Bukin 6: valley with discontinuous minimum valley
#minimum at f(-10,1) = 0
class bukin6(function_interface):
    def __init__(self):
        self.n_dim = 2
        self.bounds = [[-15,-3],[-5,3]]

    def evaluate(self,var):
        function_interface.evaluate(self,var)
        return 100*np.sqrt(abs(var[1] - 0.01*var[0]**2)) + 0.01*abs(var[0]+10)

#Eggholder: 'chaotic function' with many valleys
#minimum at f(512,404.2319) = -959.6407
class eggholder(function_interface):
    def __init__(self):
        self.n_dim = 2
        self.bounds = [[-512]*2,[512]*2]
    
    def evaluate(self,var):
        function_interface.evaluate(self,var)
        term0 = -(var[1] + 47)*np.sin(np.sqrt(abs(var[0]/2 + (var[1] + 47))))
        term1 = -var[0]*np.sin(np.sqrt(abs(var[0] - var[1]+47)))
        return term0 + term1

#cross_in_tray: trays with a cross through x=0 and y=0 lines.
#minimum at f(+- 1.34941, +-1.34941) = -2.06261
class cross_in_tray(function_interface):
    def __init__(self):
        self.n_dim = 2
        self.bounds = [[-10]*2,[10]*2]
    
    def evaluate(self,var):
        function_interface.evaluate(self,var)
        exp_factor = np.exp(abs(100 - np.sqrt(var[0]**2 + var[1]**2)/np.pi))
        internal = abs(np.sin(var[0])*np.sin(var[1])*exp_factor) + 1
        return -0.0001*internal**0.1