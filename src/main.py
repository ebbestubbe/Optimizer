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
    '''
    optfunc = Functions.test_functions.rosenbrock()  
    start_point = np.zeros(optfunc.n_dim)+3
    start_point = np.array([-1,1])
    '''
    target = io.imread('Tricasso\MonaLisa.png')
    optfunc = tricasso_func(target)
    start_point = np.array([50, 200, 200, 50,50,200,100,200,100])
    before_opt = trial_init(target)
    add_triangle(before_opt,vec = start_point)
    io.imsave('before.png',before_opt)
    
	solver = simplex(optfunc,start_point,rel_tol = 0.0000001, abs_tol = 0.0001, max_iter = 300)
    
    observer_step_log = Solvers.observers.observer_simplex_step_log(solver)
    observer_time = Solvers.observers.observer_timeit()
    
    solver.attach(observer_step_log)
    solver.attach(observer_time)
    
    val,var = solver.solve()
    
    solvetime = observer_time.get_solvetime()
    steptimes = observer_time.get_steptimes()
    
    results = observer_step_log.get_result()
    
    
    
    plt.figure(1)
    for j in range(len(results[0][1])):
        plt.plot([results[i][1][j] for i in range(len(results))])
    plt.show()
    
    plt.figure(2)
    plt.plot(steptimes)
    plt.show()
    
    plt.figure(3)
    plt.hist(steptimes)
    plt.show()
    
    print("val:")
    print(val)
    print("var:")
    print(var)
    print("solvetime: " + str(solvetime))
    print("avg steptimes: " + str(np.mean(steptimes)))
    
    '''
    product = trial_init(target)
    
    add_triangle(product,vec = var)
    io.imsave('Product.png',product)
    '''
    
if __name__ == '__main__':
    main()