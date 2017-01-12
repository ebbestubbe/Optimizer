'''
Created on 24/10/2016

@author: Ebbe
'''
import numpy as np
import matplotlib.pyplot as plt
class function_interface(object):
    '''
    n_dim: number of dimensions
    bounds: [lower,upper]
    '''    
    
    def evaluate(self,var):
        self.checkbounds(var)
        assert(self.n_dim == len(var))
    
    def checkbounds(self,var):
        #print("checking: " +str(var))
        for i in range(len(var)):
            assert(var[i] >= self.bounds[0][i] and var[i] <= self.bounds[1][i])
    
    #Call the contour plotting function with the number of points in x and y defined
    def contour(self,lower_bounds,upper_bounds,points=100,N=10):
        x = np.linspace(lower_bounds[0], upper_bounds[0],num = points)
        y = np.linspace(lower_bounds[1], upper_bounds[1],num = points)
        #X and Y are both defined at all points in the grid, resulting in redundant rows/columns 
        X,Y = np.meshgrid(x,y) 
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                #print("(" + str(X[i,j]) + ", " + str(Y[i,j]) + ")")
                Z[i,j] = self.evaluate(np.array([X[i,j],Y[i,j]]))
        plt.contour(X,Y,Z,N)
        plt.axes().set_aspect('equal', 'datalim')