# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:45:09 2016

@author: Ebbe
"""
from skimage import draw
import numpy as np

#img: img(np array) to add triangle to
#r:     3-vector rows of the corners in the triangle
#c:     3-vector columns of the corners in the triangle
#color: r-vector rgb
#NB: DRAWS THE TRIANGLE REGARDLESS OF BOUNDS IN SHAPE OR COLOR
#ASSUME THE TRIANGLE IS VALID
def add_triangle(img,r=None,c=None,color=None,vec = None):
    if(vec != None):
        r = vec[0:3]
        c = vec[3:6]
        color = vec[6:9]
    rr,cc = draw.polygon(r,c)
    img[rr,cc] = color

def add_triangles(img,vec):
    for i in range(len(vec)//9):
        ind = i*9
        r = vec[ind+0:ind+3]
        c = vec[ind+3:ind+6]
        color = vec[ind+6:ind+9]
        rr,cc = draw.polygon(r,c)
        img[rr,cc] = color
           
def add_triangles_additive_bw(img,vec):
    for i in range(len(vec)//7):
        ind = i*7
        r = vec[ind+0:ind+3]
        c = vec[ind+3:ind+6]
        color = vec[ind+6]
        rr,cc = draw.polygon(r,c)
        #img[rr,cc] += color
        img[rr,cc] = np.minimum(1.0,img[rr,cc] + color/255)
        
           
#initializ a white rgb image with the same dimensions  
def trial_init(img_target):
    #trial = np.zeros(img_target.shape,dtype=np.uint8)-1
    trial = np.zeros(img_target.shape,dtype=np.float64)
    return trial
    
#returns r,c,color of a random triangle, respecting bounds in the image and in 8-bit triplet colors
def rand_triangle(imgshape):
    r = np.random.randint(0,high = imgshape[0],size = 3)
    c = np.random.randint(0,high = imgshape[1],size = 3)
    color = np.random.randint(0,high = 256,size = 3,dtype = np.uint8)
    return r,c,color
    
#returns the sum of the euclidian color distance
def calc_cost(target,trial):
    target_64 = target.astype(np.int64)
    trial_64 = trial.astype(np.int64)
    
    difference = target_64-trial_64
    
    cost_r = np.power(difference[:,:,0],2)
    cost_g = np.power(difference[:,:,1],2)
    cost_b = np.power(difference[:,:,2],2)
    
    cost = cost_r + cost_g + cost_b
    
    costsum = np.sum(cost)
    return costsum
#returns the sum of the euclidian color distance
def calc_cost_bw(target,trial):
    #target_64 = target.astype(np.int64)
    #trial_64 = trial.astype(np.int64)
        
    difference = target-trial
        
    cost = np.power(difference,2)
    
    costsum = np.sum(cost)
    return costsum