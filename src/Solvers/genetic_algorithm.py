# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 00:11:00 2017

@author: Ebbe
"""

import numpy as np
from solver import solver_interface 
import random
class genetic_algorithm(solver_interface):
    def __init__(self,pop_size, termination_strategies = []):
        #Set func as internal variable via super constructor, maybe put more stuff into super constructor later?
        super().__init__(termination_strategies)
        self.pop_size = pop_size
        self.population = []
        self.population_orientation = 'ROW' #individuals are row vectors
        #self.n_select = np.floor(pop_size/2)
        self.id = "GENETIC_ALGORITHM"
        
    def step_alg(self):
        #Make a child for each individual in the population
        children = []
        for i in range(self.pop_size):
            parents = random.sample(range(self.pop_size),2)
            child = self.create_child(self.population[parents[0]],self.population[parents[1]])
            child_val = self.func.evaluate(child)
            children.append([child,child_val])
            
        for i in range(self.pop_size):
            self.insertpoint(children[i][0],children[i][1])
        
        self.cull_population()
        self.bestpoints.append(self.population[0])
        self.bestvalues.append(self.population_values[0])
        #print(self.population)
    #figure out a way to incoorporate point_start
    def solve_alg(self,func,point_start):
        self.func = func
        self.generate_init_pop()
        
        self.it = 0
        #keep going until a termination strategy tells the solver to fuck off
        while(True):
            self.step()
            break_bools = [i.check_termination(solver=self) for i in self.termination_strategies]
            #print(break_bools)            
            if(any(break_bools)):
                break
            
            self.it+=1
        
        return(self.bestvalues[-1],self.bestpoints[-1])
    
    def create_child(self,v1,v2):
        mask = np.random.randint(2,size = self.func.n_dim)
        mask = mask.astype(bool)
        v1_copy = np.copy(v1)
        v2_copy = np.copy(v2)
        v1_copy[~mask] = 0
        
        v2_copy[mask] = 0
        v_child =  v1_copy +  v2_copy    
        
        #if(random.uniform(0,1) > 0.9):
        mut_term = np.random.normal(0.0,0.05,self.func.n_dim)
        v_child += mut_term
        v_child = np.clip(v_child,self.func.bounds[0],self.func.bounds[1])
        return v_child
    
    def cull_population(self):
        self.population = self.population[0:self.pop_size,:]
        self.population_values = self.population_values[0:self.pop_size]
        

    #generate initial random population and evaluate
    def generate_init_pop(self):
        self.population =  np.array([np.random.uniform(self.func.bounds[0],self.func.bounds[1]) for i in range(self.pop_size)])
        #print(self.population)
        self.sortpopulation()
        self.bestpoints = [self.population[0]]
        self.bestvalues = [self.population_values[0]]
       
    def sortpopulation(self):   
        self.population_values = np.array([self.func.evaluate(self.population[i]) for i in range(len(self.population))])
        a = self.population_values.argsort()        
        self.population_values = self.population_values[a]
        self.population = self.population[a]
    
    #Method to insert a point into the self.population and self.population_values, such that the list remains sorted
    #Saves computation time, as all points do not have to be re-evaluated
    def insertpoint(self,point,value):
        #Remove the worst point ( for simplex, for GA: just insert, such that the list grows and is later culled)
        #self.population = self.population[0:-1,:]
        #self.population_values = self.population_values[0:-1]
        
        #Figure out where to put the point:
        insert_ind = np.searchsorted(self.population_values,value)
        #Insert the point and value at the sorted position        
        self.population_values = np.insert(self.population_values,insert_ind,value)
        self.population = np.insert(self.population,insert_ind,point,axis=0)
        