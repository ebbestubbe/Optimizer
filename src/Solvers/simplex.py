'''
Created on 06/11/2016

@author: Ebbe
'''
import numpy as np
from solver import solver_interface 
class simplex(solver_interface):
    
    def __init__(self, max_iter = 1000,start_size = 0.005, termination_strategies = []):
        #Set func as internal variable via super constructor, maybe put more stuff into super constructor later?
        super().__init__(termination_strategies)
        self.id = "NELDER_MEAD_SIMPLEX"
        self.start_size = start_size        

        #set tolerances and max number of iterations        
        self.max_iter = max_iter
        
        #set simplex parameter values        
        self.alpha = 1        
        self.gamma = 2
        self.rho = 0.5
        self.sigma = 0.5
        
        
    #Stepping algorithm, as per wikipedia
    def step_alg(self):
        
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
                p1 = self.points[0,:] * (1-self.sigma) #precalc
                
                for i in range(1,self.points.shape[0]):
                     self.points[i,:] = np.clip(p1 + self.sigma*self.points[i,:],self.func.bounds[0],self.func.bounds[1])
                self.sortsimplex()
        
    def solve_alg(self,func,point_start):
        
        self.func = func
        self.points = np.zeros([func.n_dim+1,func.n_dim])
        self.points[0,:] = point_start
        
        #Make all the points in the simplex(n_dim + 1)
        for i in range(func.n_dim):
            point_new = np.zeros(func.n_dim)
            point_new[i] = self.start_size
            cand_point = point_start + point_new
            self.points[i+1,:] = np.clip(cand_point,self.func.bounds[0],self.func.bounds[1])            

        self.sortsimplex()
        self.it = 0
        self.bestvals = [self.values[0]]
        
        while(self.it < self.max_iter):    
            self.step()
            self.bestvals.append(self.values[0])
            break_bools = [i.check_termination(solver=self) for i in self.termination_strategies]
            if(break_bools[0]):
                break
            '''
            to_checkwith = it - self.func.n_dim*4 #when checking convergence
            
            if(to_checkwith > 0):
                abs_diff = bestvals[to_checkwith] - bestvals[-1]
                rel_diff = abs((bestvals[to_checkwith] - bestvals[-1])/bestvals[-1])              
                abs_break = (abs_diff < self.abs_tol)
                rel_break = (rel_diff < self.rel_tol)
                
                if(abs_break or rel_break):
                    break
            '''
            self.it+=1
        
        return(self.values[0],self.points[0,:])
    
    #Sort points and values such that the lowest values come first.    
    #Should only be called when all points are to be evaluated(init and shrinking)
    def sortsimplex(self):   
        self.values = np.array([self.func.evaluate(self.points[i]) for i in range(len(self.points))])
        a = self.values.argsort()        
        self.values = self.values[a]
        self.points = self.points[a]
        
    #Explicitly only used for the simplex:
    #Method to insert a point into the self.points and self.values, such that the list remains sorted
    #Saves computation time, as all points do not have to be re-evaluated
    def insertpoint(self,point,value):
        #Remove the worst point
        self.points = self.points[0:-1,:]
        self.values = self.values[0:-1]
        #Figure out where to put the point:
        insert_ind = np.searchsorted(self.values,value)
        #Insert the point and value at the sorted position        
        self.values = np.insert(self.values,insert_ind,value)
        self.points = np.insert(self.points,insert_ind,point,axis=0)
    