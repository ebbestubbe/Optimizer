'''
Created on 06/11/2016

@author: Ebbe
'''
import matplotlib.pyplot as plt

import numpy as np

class simplex(object):
    '''
    self.points: (n_dim+1) points acting as corners in the simplex
    '''
    
    def __init__(self, func,point_start, abs_tol = 0.0001, rel_tol = 0.0001, max_iter = 1000):
        #set func and points as internal values
        self.func = func
        self.points = np.zeros([func.n_dim+1,func.n_dim])
        self.points[0,:] = point_start
                        
        #set tolerances        
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.max_iter = max_iter
        
        #set simplex parameter values        
        self.alpha = 1        
        self.gamma = 2
        self.rho = 0.5
        self.sigma = 0.5
        
        self.observers = []
        for i in range(func.n_dim):
            point_new = np.zeros(func.n_dim)
            point_new[i] = 0.05
            cand_point = point_start + point_new
            self.points[i+1,:] = np.clip(cand_point,self.func.bounds[0],self.func.bounds[1])        
    
    def attach(self,observer):
        self.observers.append(observer)
    
    def step(self):
        [i.notify_step() for i in self.observers]
        
        #evaluate all points
        values = self.sortsimplex()
        centroid = np.average(self.points[0:-1,:],axis=0)

        #3:reflection
        reflected = np.clip(centroid*(1+self.alpha) - self.alpha*self.points[-1,:],self.func.bounds[0],self.func.bounds[1])        
        reflected_val = self.func.evaluate(reflected)        
        if(values[0] <= reflected_val and reflected_val < values[-2]):
            [i.notify_reflecting() for i in self.observers]
            self.points[-1,:] = reflected
            return
        #4:expansion:
        elif(reflected_val < values[0]):
            expanded = np.clip(centroid * (1 - self.gamma) + self.gamma*reflected,self.func.bounds[0],self.func.bounds[1])
            expanded_val = self.func.evaluate(expanded)
                
            if(expanded_val< reflected_val):
                [i.notify_expanding() for i in self.observers]
                self.points[-1,:] = expanded
                return
            else:
                [i.notify_reflecting() for i in self.observers]
                self.points[-1,:] = reflected
                return
        #5:contraction
        else:
            #print("contracting")
            contracted = np.clip(centroid*(1 - self.rho) + self.rho*self.points[-1,:],self.func.bounds[0],self.func.bounds[1])
            contracted_val = self.func.evaluate(contracted)
            #print("contracted:       " + str(contracted))
            #print("contracted_val:   " + str(contracted_val))        

            if(contracted_val < values[-1]):
                [i.notify_contracting() for i in self.observers]
                self.points[-1,:] = contracted
                return
            #6:shrink
            else:
                [i.notify_shrinking() for i in self.observers]
                p1 = self.points[0,:] * (self.sigma-1) #precalc
                for i in range(1,self.points.shape[0]):
                     self.points[i,:] = np.clip(p1 + self.sigma*self.points[i,:],self.func.bounds[0],self.func.bounds[1])
    '''
    def sortsimplex(self):   
        values = np.array([self.func.evaluate(self.points[i]) for i in range(len(self.points))])
        #sort points and values such that the lowest values come first.    
        a = values.argsort()        
        values = values[a]
        self.points = self.points[a]
        return values
    '''
    def solve(self):
        [i.notify_solve() for i in self.observers]
        bestpoint = np.zeros([self.max_iter+1,self.func.n_dim])
        bestpoint[0,:] = self.points[0,:]
        values = self.sortsimplex()
        bestvals = [values[0]]
        it = 0
        while(it < self.max_iter):    
            self.step()
            values = self.sortsimplex()
            bestvals.append(values[0])
            bestpoint[it+1,:] = self.points[0,:]
            to_checkwith = it - self.func.n_dim*4            
            
            if(to_checkwith > 0):
                abs_break = (bestvals[to_checkwith] - bestvals[-1] <  self.abs_tol)
                rel_break = ((bestvals[to_checkwith] - bestvals[-1])/bestvals[-1] <  self.rel_tol)
                
                if(abs_break or rel_break):
                    print("breaking: ")
                    print("current val: " + str(bestvals[-1]))
                    print("prev val:    " + str(bestvals[to_checkwith]))
                    print("abs diff:    " + str(bestvals[to_checkwith] - bestvals[-1]))
                    print("rel diff:    " + str((bestvals[to_checkwith] - bestvals[-1])/bestvals[-1]))
                    break
            it+=1
        plt.figure(1)
        plt.plot(bestvals)
        plt.show()
        
        #plt.figure(2)
        #plt.plot(bestpoint[:,0],bestpoint[:,1],'b.')
        #plt.axis([min(bestpoint[:,0]),max(bestpoint[:,0]),min(bestpoint[:,1]),max(bestpoint[:,1])])
        #plt.show()
        
        return(values[0],self.points[0,:])
        #return(self.func.evaluate(self.points[0,:]),self.points[0,:])