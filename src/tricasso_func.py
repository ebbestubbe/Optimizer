# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 17:10:10 2016

@author: Ebbe
"""

import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage import draw

from Tricasso.tricasso import trial_init
from Tricasso.tricasso import add_triangle
from Tricasso.tricasso import add_triangles
from Tricasso.tricasso import add_triangles_additive_bw
from Tricasso.tricasso import calc_cost
from Tricasso.tricasso import calc_cost_bw

import numpy as np
from function import function_interface

class tricasso_func_set_rgb(function_interface):
    
    def __init__(self,image,n_triangles):
        super().__init__()
        self.n_dim = n_triangles*9
        self.target = image
        self.trial = trial_init(self.target)
        upper = []
        lower = []
        #Assume each triangle has the same boundaries        
        for i in range(n_triangles):
            upper.extend([self.target.shape[0]]*3)
            upper.extend([self.target.shape[1]]*3)
            upper.extend([255]*3)
            lower.extend([0]*9)
        self.bounds = [lower, upper]
       
    def evaluate(self,var):
        super(tricasso_func_set_rgb,self).evaluate(var)
        trial_copy = np.copy(self.trial)
        add_triangles(trial_copy,var)
        '''
        for i in range(len(var)//9):    
            ind = i*9
            r = var[ind + 0:ind + 3]
            c = var[ind + 3:ind + 6]
            color = var[ind + 6:ind + 9]
            add_triangle(trial_copy,r,c,color)
        '''
        cost = calc_cost(self.target,trial_copy)
        return cost
    
class tricasso_func_add_bw(function_interface):
    
    def __init__(self,image,n_triangles):
        super().__init__()
        self.n_dim = n_triangles*7
        self.target = image
        self.trial = trial_init(self.target)
        upper = []
        lower = []
        #Assume each triangle has the same boundaries        
        for i in range(n_triangles):
            upper.extend([self.target.shape[0]-1]*3)
            upper.extend([self.target.shape[1]-1]*3)
            upper.extend([255]*1)
            lower.extend([0]*7)
        self.bounds = [lower, upper]
       
    def evaluate(self,var):
        super(tricasso_func_add_bw,self).evaluate(var)
        trial_copy = np.copy(self.trial)
        add_triangles_additive_bw(trial_copy,var)
        '''
        for i in range(len(var)//9):    
            ind = i*9
            r = var[ind + 0:ind + 3]
            c = var[ind + 3:ind + 6]
            color = var[ind + 6:ind + 9]
            add_triangle(trial_copy,r,c,color)
        '''
        cost = calc_cost_bw(self.target,trial_copy)
        return cost