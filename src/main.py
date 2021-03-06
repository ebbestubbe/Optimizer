'''
Created on 24/10/2016

@author: Ebbe
'''
import numpy as np
import matplotlib.pyplot as plt

from Solvers.simplex import simplex
from Solvers.naive_line_search import naive_line_search
import Solvers.observers
from skimage import io

#from Functions.test_functions import function_sphere
import Functions.test_functions
from tricasso_func import tricasso_func
from tricasso_func import trial_init
from Tricasso.tricasso import add_triangle
from Tricasso.tricasso import add_triangles

from Solvers.terminationstrat import termination_strategy_tolerance
from Solvers.terminationstrat import termination_strategy_max_iter
from Solvers.terminationstrat import termination_strategy_max_eval
from Solvers.genetic_algorithm import genetic_algorithm


def main():
    tricasso_GA()
#    tricasso_linesearch()

def tricasso_GA():
    plt.close("all")
    resultfolder = 'Results\\'
    '''
    optfunc = Functions.test_functions.rosenbrock()  
    start_point = np.zeros(optfunc.n_dim)+3
    start_point = np.array([-1,1])
    '''
    target = io.imread('Tricasso\MonaLisa.png')
    #target = io.imread('Tricasso\shapes.png')
    #target = io.imread('Tricasso\fruit.png')
    
    optfunc = tricasso_func(target,n_triangles = 15)
    
    rel_tol = 10e-10
    abs_tol = 10e6
    
    check_depth = 10
    #rel_tol = 0
    #abs_tol = 0
    max_eval = 10000
    max_iter = 40
    t_strat_tol = termination_strategy_tolerance(rel_tol = rel_tol, abs_tol = abs_tol, check_depth = check_depth)
    t_strat_max_iter = termination_strategy_max_iter(max_iter = max_iter)
    t_strat_max_eval = termination_strategy_max_eval(max_eval = max_eval)
    termination_strategies = [t_strat_tol,t_strat_max_iter,t_strat_max_eval]
    
    pop_size = 500
    solver = genetic_algorithm(pop_size = pop_size,termination_strategies = termination_strategies)    
    
    '''
    start_point = np.array([ 100,250,250,100,100,200,200, 50, 50,#])
                             100,100,150,200,250,250,150,150,150,#])
                              50, 78,120,157,131,178, 50, 50,250,#])
                             300,350,350,350,350,300,250, 50, 50])
    '''
    start_point = np.zeros(len(optfunc.bounds[0]))
    
    for i in range(len(optfunc.bounds[0])):
        p = np.random.randint(optfunc.bounds[0][i],optfunc.bounds[1][i]+1)
        start_point[i] = p
    
    before_opt = trial_init(target)
    add_triangles(before_opt,vec = start_point)
    io.imsave(resultfolder + 'GA_before_rand.png',before_opt)

    observer_time = Solvers.observers.observer_timeit()
    observer_step_log = Solvers.observers.observer_step_log(solver)
    n_step_interval = 5
    observer_print_log = Solvers.observers.observer_step_print(solver,n_step_interval = n_step_interval)
    
    #observer_log = Solvers.observers.observer_simplex_print_log()
    
    #solver.attach(observer_step_log)
    solver.attach(observer_time)
    solver.attach(observer_step_log)
    solver.attach(observer_print_log)
    #solver.attach(observer_log)
    
    val,var = solver.solve(optfunc,start_point)

    solvetime = observer_time.get_solvetime()
    steptimes = observer_time.get_steptimes()
    
    result_log = observer_step_log.get_result()
    
    plt.figure(1)
    plt.plot([result_log[i][2] for i in range(len(result_log))],[result_log[i][0] for i in range(len(result_log))])
    plt.xlabel('Number of function evaluations')
    plt.ylabel('Function value')
    
    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(steptimes)
    
    plt.subplot(2,1,2)
    plt.hist(steptimes)
    plt.show()
    
    print("val:")
    print(val)
#    print("var:")
#    print(var)
    print("solvetime: " + str(solvetime))
    print("avg steptimes: " + str(np.mean(steptimes)))
    
    product = trial_init(target)
    
    add_triangles(product,vec = var)    
    io.imsave(resultfolder + 'GA_Product_rand.png',product)
    
def tricasso_linesearch():
    plt.close("all")
    resultfolder = 'Results\\'
    '''
    optfunc = Functions.test_functions.rosenbrock()  
    start_point = np.zeros(optfunc.n_dim)+3
    start_point = np.array([-1,1])
    '''
    target = io.imread('Tricasso\MonaLisa.png')
    #target = io.imread('Tricasso\shapes.png')
    #target = io.imread('Tricasso\fruit.png')
    
    optfunc = tricasso_func(target,n_triangles = 1)
    '''
    start_point = np.array([ 100,250,250,100,100,200,200, 50, 50,#])
                             100,100,150,200,250,250,150,150,150,#])
                              50, 78,120,157,131,178, 50, 50,250,#])
                             300,350,350,350,350,300,250, 50, 50])
    '''
    start_point = np.zeros(len(optfunc.bounds[0]))
    
    for i in range(len(optfunc.bounds[0])):
        p = np.random.randint(optfunc.bounds[0][i],optfunc.bounds[1][i]+1)
        start_point[i] = p
    
    before_opt = trial_init(target)
    add_triangles(before_opt,vec = start_point)
    io.imsave(resultfolder + 'linesearch_before_rand.png',before_opt)
    rel_tol = 10e-10
    abs_tol = 10e5
    max_iter = 100
    start_size = 1
    solver = naive_line_search(rel_tol = rel_tol, abs_tol = abs_tol, max_iter = max_iter,start_size = start_size)
    
    observer_step_log = Solvers.observers.observer_simplex_step_log(solver)
    observer_time = Solvers.observers.observer_timeit()
        
    solver.attach(observer_step_log)
    solver.attach(observer_time)
    
    val,var = solver.solve(optfunc,start_point)
    
    solvetime = observer_time.get_solvetime()
    steptimes = observer_time.get_steptimes()
    
    results = observer_step_log.get_result()
    
    plt.figure(1)
    plt.plot([results[i][1] for i in range(len(results))])
    plt.show()
    
    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(steptimes)
    
    plt.subplot(2,1,2)
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
    io.imsave(resultfolder + 'linesearch_Product_rand.png',product)

def tricasso_simplex():
    plt.close("all")
    resultfolder = 'Results\\'
    '''
    optfunc = Functions.test_functions.rosenbrock()  
    start_point = np.zeros(optfunc.n_dim)+3
    start_point = np.array([-1,1])
    '''
    #target = io.imread('Tricasso\MonaLisa.png')
    target = io.imread('Tricasso\shapes.png')

    optfunc = tricasso_func(target,n_triangles = 2)
    
    start_point = np.array([ 100,250,250,100,100,200,200, 50, 50,#])
                             100,100,150,200,250,250,150,150,150])
                            # 50, 78,120,157,131,178, 50, 50,250])#,
                            #300,350,350,350,350,300,250, 50, 50])
    
    before_opt = trial_init(target)
    add_triangles(before_opt,vec = start_point)
    io.imsave(resultfolder + 'before2.png',before_opt)
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
    plt.subplot(2,1,1)
    plt.plot(steptimes)
    
    plt.subplot(2,1,2)
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
    io.imsave(resultfolder + 'Product2.png',product)
    
if __name__ == '__main__':
    main()