'''
Created on 05/12/2016

@author: Ebbe
'''
import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage import draw

import tricasso

import os

def main():
    target = io.imread('MonaLisa.png')
    
    trial = tricasso.trial_init(target)
    costs = []
    bestcost = tricasso.calc_cost(target,trial)    
    for i in range(20):
        trial_copy = np.copy(trial)
        r1,c1,color1 = tricasso.rand_triangle(target.shape)
        tricasso.add_triangle(trial_copy,r1,c1,color1)
        cost = tricasso.calc_cost(target,trial_copy)
        if(cost<bestcost):
            bestcost = cost
            costs.append(cost)
            print("cost at " + str(i) + ": " +  str(cost))
            io.imsave('trials/trial' + str(i) + '.png',trial_copy)
    plt.plot(costs)
    plt.show()
    #io.imsave('trial.png',trial)
    
if __name__ == '__main__':
    plt.close("all")
    os.system('cls')
    main()
    #main()
    #main2()
    #main3()