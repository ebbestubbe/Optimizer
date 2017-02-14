# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 15:43:43 2017

@author: Ebbe
"""

import numpy as np
from solver import solver_interface 

class CMA_ES(solver_interface):
    def __init__(self,pop_size, termination_strategies = []):
        #Set func as internal variable via super constructor, maybe put more stuff into super constructor later?
        super().__init__(termination_strategies)
        self._lambda = pop_size
        self.population = []
        self.population_orientation = 'COLUMN' # the individual members are oriented as column vectors
        #self.n_select = np.floor(pop_size/2)
        self.id = "CMA_ES"
        
    def step_alg(self):
        
        #sort by fitness and compute weighted mean into xmean
        self.generate_offspring()
        
        self._xold = self._xmean

        self._xmean = np.average(self.population[:,0:self._mu],axis=1,weights = self._weights)
        
        #cumulation: update evolution paths
        self.update_evolution_paths()
        
        #adapt C
        self.adapt()
        
        #decompose C into B*diag(D^2)*B'
        if(self.it - self._eigeneval  > self._lambda/(10*self.func.n_dim*(self._c_1+self._c_mu))):
            self.decompose_C()

    #figure out a way to incoorporate point_start
    def solve_alg(self,func,point_start):
        self.func = func
        #self._xmean = np.random.uniform(self.func.bounds[0],self.func.bounds[1])
        self._xmean = point_start 
        
        #user defined: figure out a way to incorporate _sigma in a good way. Maybe common interface for local, and one for global?
        self._sigma = 0.5
        
        self.init_var()
        
        self.population = np.zeros((self.func.n_dim,self._lambda))
        self.population_values = np.zeros(self._lambda)
        
        self.it = 0
        self.bestvals = []
        self.bestpointsofar = None
        self.bestvaluesofar = np.inf        

        while(True):
            self.it += 1
            
            #generate and evaluate lambda offspring
            self.step()
            '''
            print("C:")
            print(self._C)
            print("Sigma:")
            print(self._sigma)
            print("D:")
            print(self._D)
            print("B:")
            print(self._B)
            '''
            break_bools = [i.check_termination(solver=self) for i in self.termination_strategies]
            #print(break_bools)            
            if(any(break_bools)):
                break
        return (self.bestvaluesofar,self.bestpointsofar)
       
    def init_var(self):
        #SELECTION: lambda is initialized in __init__()
        self._mu = self._lambda/2.0
        self._weights = np.array([np.log(self._mu+0.5) - np.log(i+1) for i in range(int(self._mu))])
        self._mu = int(np.floor(self._mu))
        self._weights = self._weights/np.sum(self._weights)
        self._mu_eff = np.sum(self._weights)**2/np.sum(np.power(self._weights,2))
                
        #ADAPTATION
        #time constant for cumulation for C:
        self._c_c = (4 + self._mu_eff/self.func.n_dim)/(self.func.n_dim + 4 + 2*self._mu_eff/self.func.n_dim)
        #time constant for cumulation for sigma control:
        self._c_s = (self._mu_eff + 2)/(self.func.n_dim + self._mu_eff + 5)
        #Learning rate for rank-one-update
        self._c_1 = 2.0/((self.func.n_dim + 1.3)**2 + self._mu_eff)
        #Learning rate for rank-mu-update
        self._c_mu = min(1 - self._c_1, 2.0*(self._mu_eff - 2.0 + 1.0/self._mu_eff)/((self.func.n_dim + 2.0)**2 + self._mu_eff))
        #Damping for sigma(usually close to 1):
        self._d_s = 1 + 2*max(0,np.sqrt((self._mu_eff - 1)/(self.func.n_dim + 1)) - 1) + self._c_s

        #INITIALIZE DYNAMIC PARAMETERS
        
        self._path_C = np.zeros((self.func.n_dim,1))
        self._path_s = np.zeros((self.func.n_dim,1))
        
        self._B = np.identity(self.func.n_dim)
        self._D = np.ones((self.func.n_dim))
        Dsq = np.diag(np.multiply(self._D,self._D))
        #self._C = np.dot(self._B,np.dot(Dsq,np.transpose(self._B)))
        self._C = self._B.dot(Dsq).dot(np.transpose(self._B))
        #self._invsqrtC = np.dot(self._B,np.dot(np.diag(1.0/self._D),np.transpose(self._B)))
        self._invsqrtC = self._B.dot(np.diag(1.0/self._D)).dot(np.transpose(self._B))
        self._eigeneval = 0
        self._chiN = np.sqrt(self.func.n_dim)*(1 - 1.0/(4.0*self.func.n_dim) + 1.0/(21.0*self.func.n_dim**2))
        
    def generate_offspring(self):
        for i in range(self._lambda):
            badcandidate = True
            while(badcandidate):
                
                randpart = np.random.normal(size = self._D.shape)
                #candidate = self._xmean + self._sigma * np.dot(self._B,self._D*randpart)
                candidate = self._xmean + self._sigma * self._B.dot(self._D*randpart)
                withinlower = candidate>self.func.bounds[0]
                withinupper = candidate<self.func.bounds[1]
                if(withinlower.all() and withinupper.all()):
                    badcandidate = False
                    self.population[:,i] = candidate
        self.sortpopulation()
        self.setbest()
        self.bestvals.append(self.bestvalue)
        
    def update_evolution_paths(self):
        
        self._path_s = (1-self._c_s)*self._path_s + np.sqrt(self._c_s*(2 - self._c_s)*self._mu_eff) * self._invsqrtC * (self._xmean - self._xold)/self._sigma
        self._hsig = np.linalg.norm(self._path_s)/np.sqrt(1 - np.power((1 - self._c_s),(2*self.it)))/self._chiN < 1.4 + 2/(self.func.n_dim + 1)
        self._path_C = (1 - self._c_c)*self._path_C + self._hsig * np.sqrt(self._c_c*(2 - self._c_c)*self._mu_eff) * (self._xmean - self._xold)/self._sigma

    def adapt(self):
        #temporary array
        artmp = (1/self._sigma) * (self.population[:,0:self._mu] - np.tile(self._xold[:,None],(1,self._mu)))
        
        #old matrix + (rank one + minor correction if hsig != 1) + rank mu
        #self._C = (1 - self._c_1 - self._c_mu)*self._C + self._c_1 * (self._path_C.dot(np.transpose(self._path_C)) + (1-self._hsig)*self._c_c*(2-self._c_c)*self._C) + self._c_mu * np.dot(artmp,np.dot(np.diag(self._weights),np.transpose(artmp))) 
        self._C = (1 - self._c_1 - self._c_mu)*self._C + self._c_1 * (self._path_C.dot(np.transpose(self._path_C)) + (1-self._hsig)*self._c_c*(2-self._c_c)*self._C) + self._c_mu * artmp.dot(np.diag(self._weights)).dot(np.transpose(artmp)) 
        
        #adapt step size
        #self._sigma = self._sigma + np.exp((self._c_s/self._d_s) * (np.linalg.norm(self._path_s)/self._chiN - 1))

        self._sigma = self._sigma * np.exp((self._c_s/self._d_s) * (np.linalg.norm(self._path_s)/self._chiN - 1))

    def decompose_C(self):
        self._eigeneval = self.it
        self._C = np.triu(self._C) + np.transpose(np.triu(self._C,1))
        [self._D, self._B] = np.linalg.eig(self._C)
        self._D = np.sqrt(self._D)
        
        self._invsqrtC = np.dot(self._B,np.dot(np.diag(1.0/self._D),np.transpose(self._B)))
        
    def sortpopulation(self):
        #here: vectors/individuals in the population are column vectors, so we first transpose, then sort, then transpose again
        self.population = np.transpose(self.population)
        
        self.population_values = np.array([self.func.evaluate(self.population[i]) for i in range(self._lambda)])
        a = self.population_values.argsort()        
        self.population_values = self.population_values[a]
        self.population = self.population[a]
        
        #transposing back
        self.population = np.transpose(self.population)

        self.setbest()

    #Keep track of the best point in a seperate variable, for ease of observers and result handling:
    def setbest(self):
        self.bestpoint = self.population[:,0]
        self.bestvalue = self.population_values[0]
        if(self.bestvalue < self.bestvaluesofar):
            self.bestvaluesofar = self.bestvalue
            self.bestpointsofar = self.bestpoint
    #Method to insert a point into the self.population and self.population_values, such that the list remains sorted
    #Saves computation time, as all points do not have to be re-evaluated
    #def insertpoint(self,point,value):
        #Remove the worst point ( for simplex, for GA: just insert, such that the list grows)
        #self.population = self.population[0:-1,:]
        #self.population_values = self.population_values[0:-1]
        
        #Figure out where to put the point:
        #insert_ind = np.searchsorted(self.population_values,value)
        #Insert the point and value at the sorted position        
        #self.population_values = np.insert(self.population_values,insert_ind,value)
        #self.population = np.insert(self.population,insert_ind,point,axis=0)
        #self.setbest()
        
