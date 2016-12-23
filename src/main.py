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
from Tricasso.tricasso import add_triangles

def main():
    plt.close("all")
    resultfolder = 'Results\\'
    '''
    optfunc = Functions.test_functions.rosenbrock()  
    start_point = np.zeros(optfunc.n_dim)+3
    start_point = np.array([-1,1])
    '''
    target = io.imread('Tricasso\MonaLisa.png')
    target = io.imread('Tricasso\shapes.png')

    optfunc = tricasso_func(target,n_triangles = 1)
    
    start_point = np.array([ 100,250,250,100,100,200,200, 50, 50])
                             #100,100,150,200,250,250,150,150,150])
                            # 50, 78,120,157,131,178, 50, 50,250])#,
                            #300,350,350,350,350,300,250, 50, 50])
    
    before_opt = trial_init(target)
    add_triangles(before_opt,vec = start_point)
    io.imsave(resultfolder + 'before.png',before_opt)
    rel_tol = 0#0.00001
    abs_tol = 0#0.1
    max_iter = 1000
    start_size = 2
    solver = simplex(optfunc,start_point,rel_tol = rel_tol, abs_tol = abs_tol, max_iter = max_iter,start_size = start_size)
    
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
    plt.plot([results[i][1][0] for i in range(len(results))])
    plt.show()
    
    plt.figure(3)
    plt.plot(steptimes)
    plt.show()
    
    plt.figure(4)
    plt.hist(steptimes)
    plt.show()
    
    print("val:")
    print(val)
    print("var:")
    print(var)
    print("solvetime: " + str(solvetime))
    print("avg steptimes: " + str(np.mean(steptimes)))
    
    
    product = trial_init(target)
    
    add_triangles(product,vec = var)
    io.imsave(resultfolder + 'Product.png',product)
    
    
if __name__ == '__main__':
    main()