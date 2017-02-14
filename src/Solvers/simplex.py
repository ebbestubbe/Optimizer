'''
Created on 06/11/2016

@author: Ebbe
'''
import numpy as np
from solver import solver_interface 
class simplex(solver_interface):
    
    def __init__(self,start_size = 0.005, termination_strategies = []):
        #Set func as internal variable via super constructor, maybe put more stuff into super constructor later?
        super().__init__(termination_strategies)
        self.id = "NELDER_MEAD_SIMPLEX"
        self.start_size = start_size        
        self.population = []
        self.population_orientation = 'ROW' #individuals are row vectors
        #set simplex parameter values        
        self.alpha = 1        
        self.gamma = 2
        self.rho = 0.5
        self.sigma = 0.5
        
        
    #Stepping algorithm, as per wikipedia
    def step_alg(self):
        
        centroid = np.average(self.population[0:-1,:],axis=0)
        
        #3:reflection
        reflected = np.clip(centroid*(1+self.alpha) - self.alpha*self.population[-1,:],self.func.bounds[0],self.func.bounds[1])        
        reflected_val = self.func.evaluate(reflected)        
        if(self.population_values[0] <= reflected_val and reflected_val < self.population_values[-2]):
            [i.notify_reflecting() for i in self.observers]
            self.insertpoint(reflected,reflected_val)
            
        #4:expansion:
        elif(reflected_val < self.population_values[0]):
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
            contracted = np.clip(centroid*(1 - self.rho) + self.rho*self.population[-1,:],self.func.bounds[0],self.func.bounds[1])
            contracted_val = self.func.evaluate(contracted)
            
            if(contracted_val < self.population_values[-1]):
                [i.notify_contracting() for i in self.observers]
                self.insertpoint(contracted,contracted_val)
                
            #6:shrink
            else:
                [i.notify_shrinking() for i in self.observers]
                p1 = self.population[0,:] * (1-self.sigma) #precalc
                
                for i in range(1,self.population.shape[0]):
                     self.population[i,:] = np.clip(p1 + self.sigma*self.population[i,:],self.func.bounds[0],self.func.bounds[1])
                self.sortpopulation()
        
    def solve_alg(self,func,point_start):
        
        self.func = func
        self.population = np.zeros([func.n_dim+1,func.n_dim])
        self.population[0,:] = point_start
        
        #Make all the points in the simplex(n_dim + 1)
        for i in range(func.n_dim):
            point_new = np.zeros(func.n_dim)
            point_new[i] = self.start_size
            cand_point = point_start + point_new
            self.population[i+1,:] = np.clip(cand_point,self.func.bounds[0],self.func.bounds[1])            

        self.sortpopulation()
        self.it = 0
        self.bestvals = [self.population_values[0]]
        #keep going until a termination strategy tells the solver to fuck off
        while(True):
            
            self.step()
            self.bestvals.append(self.population_values[0])
            break_bools = [i.check_termination(solver=self) for i in self.termination_strategies]
            #print(break_bools)            
            if(any(break_bools)):
                break
            
            self.it+=1
        
        return(self.population_values[0],self.population[0,:])
    
    #Sort points and values such that the lowest values come first.    
    #Should only be called when all points are to be evaluated(init and shrinking)
    def sortpopulation(self):   
        self.population_values = np.array([self.func.evaluate(self.population[i]) for i in range(len(self.population))])
        a = self.population_values.argsort()        
        self.population_values = self.population_values[a]
        self.population = self.population[a]
        self.setbest()
        
    #Explicitly only used for the simplex:
    #Method to insert a point into the self.population and self.population_values, such that the list remains sorted
    #Saves computation time, as all points do not have to be re-evaluated
    def insertpoint(self,point,value):
        #Remove the worst point
        self.population = self.population[0:-1,:]
        self.population_values = self.population_values[0:-1]
        #Figure out where to put the point:
        insert_ind = np.searchsorted(self.population_values,value)
        #Insert the point and value at the sorted position        
        self.population_values = np.insert(self.population_values,insert_ind,value)
        self.population = np.insert(self.population,insert_ind,point,axis=0)
        self.setbest()
        
#Keep track of the best point in a seperate variable, for ease of observers and result handling:
    def setbest(self):
        self.bestpoint = self.population[0,:]
        self.bestvalue = self.population_values[0]