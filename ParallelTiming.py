import ClarkApprox
import argparse
import numpy as np
import math
from numba import njit
import time
import matplotlib.pyplot as plt
import PolyLog
#import MPMathIntegral
import RationalApprox
import Zimmerman
import Goldin
import Quadrature
from matplotlib.legend_handler import HandlerLine2D


def time_function(planck_formula, num_repeats=2):
    """
    Time the function using the time module.
    
    Parameters:
        planck_formula (function): The function to time.
        num_repeats (int): The number of times to repeat the timing.
    
    Returns:
        float: The average time taken to run the function.
    """
    n_groups = 20
    n_cells = 100000
    n = (n_groups)*n_cells
    x = np.zeros((n_groups+1)*n_cells)
    T = np.linspace(0.1, 4, n_cells)
    #for i in range(n_cells):
    #    x[(i*n_groups+1):((i+1)*n_groups+1)] = np.logspace(-1,math.log10(20),n_groups)/T[i]
    x = np.logspace(-1,math.log10(20),n_groups*n_cells)
    total_time = 0.0
    tmp = planck_formula(x, n)
    times = np.zeros(num_repeats)
    for _ in range(num_repeats):
        start_time = time.time()
        planck_formula(x, n)
        end_time = time.time()
        times[_] = (end_time - start_time)/n
        total_time += end_time - start_time
    return times #total_time / num_repeats / n

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timing code for ClarkApprox functions")
    parser.add_argument('--num_repeats', type=int, default=10, help='Number of repetitions for timing')
    
    args = parser.parse_args()

    num_repeats = args.num_repeats
    # Time the functions
    f = lambda x,n: ClarkApprox.clark_formula_parallel(x,n,N=21,L=10)
    clark_time = time_function(f, num_repeats)
    print(f"Clark time 21-10: {np.mean(clark_time)}")
    print(f"Clark time 21-10: {np.max(clark_time)}")
    print(f"Clark time 21-10: {np.min(clark_time)}")
    f = lambda x,n: ClarkApprox.clark_formula_parallel(x,n,N=9,L=3)
    clark_time93 = time_function(f, num_repeats)
    #print(f"Clark time 9-3: {np.mean(clark_time93)}")
    f = lambda x,n: PolyLog.b_PolyParallel(x,n)
    poly_time = time_function(f, num_repeats)
    #print(f"PolyLog time: {poly_time}")
    f = lambda x,n: RationalApprox.bRationalParallel(x,n)
    rational_time = time_function(f, num_repeats)
    #print(f"Rational time: {rational_time}")
    f = lambda x,n: Zimmerman.zimIntParallel(x,n)
    zimmerman_time = time_function(f, num_repeats)
    #print(f"Zimmerman time: {zimmerman_time}")
    f = lambda x,n: Goldin.bGoldinParallel(x,n)
    goldin_time = time_function(f, num_repeats)
    #print(f"Goldin time: {goldin_time}")
    f = lambda x,n: Quadrature.gl_parallel(x,n,4,*np.polynomial.legendre.leggauss(2))
    quad_time4 = time_function(f, num_repeats)
    #print(f"Quadrature time n=2: {quad_time}")
    f = lambda x,n: Quadrature.gl_parallel(x,n,8,*np.polynomial.legendre.leggauss(10))
    quad_time8 = time_function(f, num_repeats)
    #print(f"Quadrature time n=16: {quad_time}")
    f = lambda x,n: Quadrature.gl_parallel(x,n,16,*np.polynomial.legendre.leggauss(20))
    quad_time16 = time_function(f, num_repeats)
    #print(f"Quadrature time n=20: {quad_time}")
    #f = lambda x,n: MPMathIntegral.compute_PiParallel(x,n)
    #mp_time = time_function(f, num_repeats)
    #print(f"MPMath time: {mp_time}")
    # Plot the results
    fig, ax = plt.subplots()
    plt.boxplot([clark_time, clark_time93, poly_time, rational_time, zimmerman_time, goldin_time, quad_time4, quad_time8, quad_time16], tick_labels=["Clark 21-10", "Clark 9-3", "PolyLog", "Rational", "Zimmerman", "Goldin", "Quadrature n=4", "Quadrature n=8", "Quadrature n=16"])
    plt.yscale('log')
    plt.ylabel("Time per function call (s)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("timing.pdf")
    plt.show()