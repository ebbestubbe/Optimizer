'''
Created on 24/10/2016

@author: Ebbe
'''
import numpy as np
import matplotlib.pyplot as plt

from Solvers.random_solver import Random_Solver
from Solvers.simplex import simplex
import Solvers.observers
from skimage import io

#from Functions.test_functions import function_sphere
import Functions.test_functions
from tricasso_func import tricasso_func
from tricasso_func import trial_init
from Tricasso.tricasso import add_triangle

def main():
    plt.close("all")  
    
    optfunc = Functions.test_functions.rosenbrock()  
    #start_point = np.zeros(optfunc.n_dim)+3
    #start_point = np.array([-1,1])
    start_point = [-1,1]
    #target = io.imread('Tricasso\MonaLisa.png')
    #optfunc = tricasso_func(target)
    #start_point = np.array([50, 200, 200, 50,50,200,100,200,100])
    #before_opt = trial_init(target)
    #add_triangle(before_opt,vec = start_point)
    #io.imsave('before.png',before_opt)
    #print(start_point)
    
    solver = simplex(optfunc,start_point,rel_tol = 0, abs_tol = 0, max_iter = 185)
    #observer1 = Solvers.observers.observer_simplex_print_log()
    
    #observer2 = Solvers.observers.observer_simplex_step_log(solver)
    #solver.attach(observer1)
    #solver.attach(observer2)
    
    val,var = solver.solve()
    print("val:")
    print(val)
    
    print("var:")
    print(var)
    #log = observer2.get_result()
    #print(log[0])
    #print(log[-1])
    
    return
    product = trial_init(target)
    
    np.round(product)
    add_triangle(product,vec = var)
    io.imsave('Product.png',product)

if __name__ == '__main__':
    main()