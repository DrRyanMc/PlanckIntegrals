import ClarkApprox
import argparse
import numpy as np
import math
from numba import njit
import time
import matplotlib.pyplot as plt
import PolyLog
import RationalApprox
import Zimmerman
import Goldin
import Quadrature
from matplotlib.legend_handler import HandlerLine2D


def time_function(planck_formula, num_repeats=100):
    """
    Times the compute_Pi_L function over a range of x values and calculates statistics.
    """
    x_values = np.logspace(-1, np.log10(40), 20) 

    execution_times = []
    start_time = time.time()
    planck_formula(0,1)  # Warm up the function for numba
    end_time = time.time()
    for x in x_values:
        times = []
        for _ in range(num_repeats):
            start_time = time.time()
            planck_formula(0,x)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Collect the mean execution time for this x
        execution_times.append({
            'x': x,
            'mean_time': np.mean(times[1:]),
            'std_time': np.std(times[1:])
        })

    return execution_times
class HandlerLineEmoji(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        line = super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)[0]
        # Add an emoji as text
        emoji = legend.get_figure().text(
            xdescent + width / 2, ydescent + height / 2, "☭",
            fontsize=fontsize, ha='center', va='center', transform=trans
        )
        return [line, emoji]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timing code for ClarkApprox functions")
    parser.add_argument('--num_repeats', type=int, default=10, help='Number of repetitions for timing')
    args = parser.parse_args()

    num_repeats = args.num_repeats


    plt.figure(figsize=(11, 6))
    #time clark
    f = lambda a,b: ClarkApprox.clark_formula(a,b,N=21,L=10)
    timing_results_Clark = time_function(f, num_repeats=num_repeats)
    # Extract data for plotting
    x_values = [result['x'] for result in timing_results_Clark]
    mean_times = [result['mean_time'] for result in timing_results_Clark]
    std_times = [result['std_time'] for result in timing_results_Clark]
    plt.loglog(x_values, mean_times, label="Clark")

    #time polylog
    f = lambda a,b: PolyLog.b_Poly(a,b)
    timing_results_Poly = time_function(f, num_repeats=num_repeats)
    x_values = [result['x'] for result in timing_results_Poly]
    mean_times = [result['mean_time'] for result in timing_results_Poly]
    std_times = [result['std_time'] for result in timing_results_Poly]
    plt.loglog(x_values, mean_times, linestyle="dashed", label="PolyLog")

    #time rational
    f = lambda a,b: RationalApprox.bRational(a,b)
    timing_results_Rational = time_function(f, num_repeats=num_repeats)
    x_values = [result['x'] for result in timing_results_Rational]
    mean_times = [result['mean_time'] for result in timing_results_Rational]
    std_times = [result['std_time'] for result in timing_results_Rational]
    plt.loglog(x_values, mean_times, linestyle="-.", label="Rational")

    #time quadrature
    n = 2
    nodes, weights = np.polynomial.legendre.leggauss(n)
    f = lambda a,b: Quadrature.gauss_legendre_quadrature(a,b,n,nodes,weights)
    timing_results_Quadrature = time_function(f, num_repeats=num_repeats)
    x_values = [result['x'] for result in timing_results_Quadrature]
    mean_times = [result['mean_time'] for result in timing_results_Quadrature]
    std_times = [result['std_time'] for result in timing_results_Quadrature]
    plt.loglog(x_values, mean_times, marker="s", label="GL n=2")
    
    #time quadrature
    n = 16
    nodes, weights = np.polynomial.legendre.leggauss(n)
    f = lambda a,b: Quadrature.gauss_legendre_quadrature(a,b,n,nodes,weights)
    timing_results_Quadrature = time_function(f, num_repeats=num_repeats)
    x_values = [result['x'] for result in timing_results_Quadrature]
    mean_times = [result['mean_time'] for result in timing_results_Quadrature]
    std_times = [result['std_time'] for result in timing_results_Quadrature]
    plt.loglog(x_values, mean_times, marker="o", label="GL n=16")

    #time zimmerman
    f = lambda a,b: Zimmerman.zimInt(a,b)
    timing_results_Zimmerman = time_function(f, num_repeats=num_repeats)
    x_values = [result['x'] for result in timing_results_Zimmerman]
    mean_times = [result['mean_time'] for result in timing_results_Zimmerman]
    std_times = [result['std_time'] for result in timing_results_Zimmerman]
    plt.loglog(x_values, mean_times, marker="^", label="Zimmerman")

    #time goldin
    f = lambda a,b: Goldin.bGoldin(a,b)
    timing_results_Goldin = time_function(f, num_repeats=num_repeats)
    x_values = [result['x'] for result in timing_results_Goldin]
    mean_times = [result['mean_time'] for result in timing_results_Goldin]
    std_times = [result['std_time'] for result in timing_results_Goldin]
    line_goldin, = plt.loglog(x_values, mean_times,color='gray', label="Goldin")
# Add emoji markers manually
    for x, y in zip(x_values, mean_times):
        plt.text(x, y, '☭', fontsize=12, ha='center', va='center')

    # Adjust layout to make room for the legend
    #plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjust the right margin to make room for the legend

    #plt.title("Execution Time of compute_Pi_L(x) Over Range of x", fontsize=16)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("Execution Time (seconds)", fontsize=14)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), handler_map={line_goldin: HandlerLineEmoji()})
    
    plt.grid()
    plt.savefig("Timing_" + str(num_repeats) + ".pdf")
    plt.show()
