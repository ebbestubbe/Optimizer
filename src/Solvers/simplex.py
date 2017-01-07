'''
Created on 06/11/2016

@author: Ebbe
'''
import numpy as np

class simplex(object):
    '''
    self.points: (n_dim+1) points acting as corners in the simplex
    '''
    
    def __init__(self, func,point_start, abs_tol = 0.0001, rel_tol = 0.0001, max_iter = 1000,start_size = 0.005):
        #set func and points as internal values
        self.func = func
        self.points = np.zeros([func.n_dim+1,func.n_dim])
        self.points[0,:] = point_start
        self.start_size = start_size        

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
            point_new[i] = self.start_size
            cand_point = point_start + point_new
            self.points[i+1,:] = np.clip(cand_point,self.func.bounds[0],self.func.bounds[1])        
    
    def attach(self,observer):
        self.observers.append(observer)
    
    def step(self):
        [i.notify_step_start() for i in self.observers]
        
        centroid = np.average(self.points[0:-1,:],axis=0)
        
        #3:reflection
        reflected = np.clip(centroid*(1+self.alpha) - self.alpha*self.points[-1,:],self.func.bounds[0],self.func.bounds[1])        
        reflected_val = self.func.evaluate(reflected)        
        if(self.values[0] <= reflected_val and reflected_val < self.values[-2]):
            [i.notify_reflecting() for i in self.observers]
            self.insertpoint(reflected,reflected_val)
            
        #4:expansion:
        elif(reflected_val < self.values[0]):
            expanded = np.clip(centroid * (1 - self.gamma) + self.gamma*reflected,self.func.bounds[0],self.func.bounds[1])
            expanded_val = self.func.evaluate(expanded)
            
            if(expanded_val< reflected_val):
                [i.notify_expanding() for i in self.observers]
                self.insertpoint(expanded,expanded_val)
                
            else:
                [i.notify_reflecting() for i in self.observers]
                self.insertpoint(reflected,reflected_val)
                
        #5:contraction
        else:
            contracted = np.clip(centroid*(1 - self.rho) + self.rho*self.points[-1,:],self.func.bounds[0],self.func.bounds[1])
            contracted_val = self.func.evaluate(contracted)
            
            if(contracted_val < self.values[-1]):
                [i.notify_contracting() for i in self.observers]
                self.insertpoint(contracted,contracted_val)
                
            #6:shrink
            else:
                [i.notify_shrinking() for i in self.observers]
                #p1 = self.points[0,:] * (self.sigma-1) #precalc
                p1 = self.points[0,:] * (1-self.sigma) #precalc
                
                for i in range(1,self.points.shape[0]):
                     self.points[i,:] = np.clip(p1 + self.sigma*self.points[i,:],self.func.bounds[0],self.func.bounds[1])
                self.sortsimplex()
        [i.notify_step_end() for i in self.observers]
        
    #Method to insert a point into the self.points and self.values
    def insertpoint(self,point,value):
        #Remove the worst point
        self.points = self.points[0:-1,:]
        self.values = self.values[0:-1]
        #Figure out where to put the point:
        insert_ind = np.searchsorted(self.values,value)
        #Insert the point and value at the sorted position        
        self.values = np.insert(self.values,insert_ind,value)
        self.points = np.insert(self.points,insert_ind,point,axis=0)
                     
    def sortsimplex(self):   
        self.values = np.array([self.func.evaluate(self.points[i]) for i in range(len(self.points))])
        #sort points and values such that the lowest values come first.    
        #to save computation time:
        #keep these as a variable within the simplex, and maintain the order on manipulation
        a = self.values.argsort()        
        self.values = self.values[a]
        self.points = self.points[a]
        
    def solve(self):
        
        [i.notify_solve_start() for i in self.observers]
        #bestpoint = np.zeros([self.max_iter+1,self.func.n_dim])
        #bestpoint[0,:] = self.points[0,:]
        self.sortsimplex()
        bestvals = [self.values[0]]
        it = 0
        while(it < self.max_iter):    
            self.step()
            #values = self.sortsimplex()
            bestvals.append(self.values[0])
            #bestpoint[it+1,:] = self.points[0,:]
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
            it+=1
        
        [i.notify_solve_end() for i in self.observers]
        return(self.values[0],self.points[0,:])