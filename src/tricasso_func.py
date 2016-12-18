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
from Tricasso.tricasso import calc_cost
import os

import numpy as np
from function import function_interface

class tricasso_func(function_interface):
    
    def __init__(self,image):
        self.n_dim = 9
        self.target = image
        self.trial = trial_init(self.target)
        upper_r = [self.target.shape[0]]*3
        upper_c = [self.target.shape[1]]*3
        upper_color = [255]*3
        upper_r.extend(upper_c)
        upper_r.extend(upper_color)        
        self.bounds = [[0]*9, upper_r]
       
    def evaluate(self,var):
        function_interface.evaluate(self,var)
        #assume 1 triangle:
        r = var[0:3]
        c = var[3:6]
        color = var[6:9]
        trial_copy = np.copy(self.trial)
        add_triangle(trial_copy,r,c,color)
        cost = calc_cost(self.target,trial_copy)
        return cost