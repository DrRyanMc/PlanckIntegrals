import ClarkApprox
import argparse
import numpy as np
import math
from numba import njit
import time
import matplotlib.pyplot as plt
import PolyLog
import MPMathIntegral
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
    results = []
    start_time = time.time()
    planck_formula(0,1)  # Warm up the function for numba
    end_time = time.time()
    for x in x_values:
        times = []
        for _ in range(num_repeats):
            start_time = time.time()
            tmp=planck_formula(0,x)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Collect the mean execution time for this x
        execution_times.append({
            'x': float(x),
            'val':float(planck_formula(0,x)),
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

    f = lambda a,b: MPMathIntegral.compute_Pi(a,b)
    timing_results_MPMath = time_function(f, num_repeats=num_repeats)
    results_True = [float(result['val']) for result in timing_results_MPMath]
    
    plt.figure(figsize=(11, 6))
    #time clark
    f = lambda a,b: ClarkApprox.clark_formula(a,b,N=21,L=10)
    timing_results_Clark = time_function(f, num_repeats=num_repeats)
    # Extract data for plotting
    x_values = [result['x'] for result in timing_results_Clark]
    mean_times = [result['mean_time'] for result in timing_results_Clark]
    std_times = [result['std_time'] for result in timing_results_Clark]
    results_Clark = [result['val'] for result in timing_results_Clark]
    errors_Clark = np.abs(np.array(results_True) - np.array(results_Clark))/np.array(results_True)
    mean_times_Clark = mean_times.copy()

    plt.loglog(x_values, mean_times, label="Clark")
    print(f"Clark mean times: {np.mean(mean_times)} std times: {np.mean(std_times)}")
    #time polylog
    f = lambda a,b: PolyLog.b_Poly(a,b)
    timing_results_Poly = time_function(f, num_repeats=num_repeats)
    x_values = [result['x'] for result in timing_results_Poly]
    mean_times = [result['mean_time'] for result in timing_results_Poly]
    std_times = [result['std_time'] for result in timing_results_Poly]
    results_Poly = [result['val'] for result in timing_results_Poly]
    errors_Poly = np.abs(np.array(results_True) - np.array(results_Poly))/np.array(results_True)
    mean_times_Poly = mean_times.copy()

    plt.loglog(x_values, mean_times, linestyle="dashed", label="PolyLog")
    print(f"PolyLog mean times: {np.mean(mean_times)} std times: {np.mean(std_times)}")
    #time rational
    f = lambda a,b: RationalApprox.bRational(a,b)
    timing_results_Rational = time_function(f, num_repeats=num_repeats)
    x_values = [result['x'] for result in timing_results_Rational]
    mean_times = [result['mean_time'] for result in timing_results_Rational]
    std_times = [result['std_time'] for result in timing_results_Rational]
    results_Rational = [result['val'] for result in timing_results_Rational]
    errors_Rational = np.abs(np.array(results_True) - np.array(results_Rational))/np.array(results_True)
    mean_times_Rational = mean_times.copy()

    plt.loglog(x_values, mean_times, linestyle="-.", label="Rational")
    print(f"Rational mean times: {np.mean(mean_times)} std times: {np.mean(std_times)}")
    #time quadrature
    n = 2
    nodes, weights = np.polynomial.legendre.leggauss(n)
    f = lambda a,b: Quadrature.gauss_legendre_quadrature(a,b,n,nodes,weights)
    timing_results_Quadrature = time_function(f, num_repeats=num_repeats)
    x_values = [result['x'] for result in timing_results_Quadrature]
    mean_times = [result['mean_time'] for result in timing_results_Quadrature]
    std_times = [result['std_time'] for result in timing_results_Quadrature]
    results_Quadrature2 = [result['val'] for result in timing_results_Quadrature]
    errors_Quadrature2 = np.abs(np.array(results_True) - np.array(results_Quadrature2))/np.array(results_True)
    mean_times_Quadrature2 = mean_times.copy()

    plt.loglog(x_values, mean_times, marker="s", label="GL n=2")
    print(f"GL n=2 mean times: {np.mean(mean_times)} std times: {np.mean(std_times)}")
    #time quadrature
    n = 16
    nodes, weights = np.polynomial.legendre.leggauss(n)
    f = lambda a,b: Quadrature.gauss_legendre_quadrature(a,b,n,nodes,weights)
    timing_results_Quadrature = time_function(f, num_repeats=num_repeats)
    x_values = [result['x'] for result in timing_results_Quadrature]
    mean_times = [result['mean_time'] for result in timing_results_Quadrature]
    std_times = [result['std_time'] for result in timing_results_Quadrature]
    results_Quadrature16 = [result['val'] for result in timing_results_Quadrature]
    errors_Quadrature16 = np.abs(np.array(results_True) - np.array(results_Quadrature16))/np.array(results_True)
    mean_times_Quadrature16 = mean_times.copy()

    plt.loglog(x_values, mean_times, marker="o", label="GL n=16")
    print(f"GL n=16 mean times: {np.mean(mean_times)} std times: {np.mean(std_times)}")
    #time zimmerman
    f = lambda a,b: Zimmerman.zimInt(a,b)
    timing_results_Zimmerman = time_function(f, num_repeats=num_repeats)
    x_values = [result['x'] for result in timing_results_Zimmerman]
    mean_times = [result['mean_time'] for result in timing_results_Zimmerman]
    std_times = [result['std_time'] for result in timing_results_Zimmerman]
    results_Zimmerman = [result['val'] for result in timing_results_Zimmerman]
    errors_Zimmerman = np.abs(np.array(results_True) - np.array(results_Zimmerman))/np.array(results_True)
    mean_times_Zimmerman = mean_times.copy()

    plt.loglog(x_values, mean_times, marker="^", label="Zimmerman")
    print(f"Zimmerman mean times: {np.mean(mean_times)} std times: {np.mean(std_times)}")
    #time goldin
    f = lambda a,b: Goldin.bGoldin(a,b)
    timing_results_Goldin = time_function(f, num_repeats=num_repeats)
    x_values = [result['x'] for result in timing_results_Goldin]
    mean_times = [result['mean_time'] for result in timing_results_Goldin]
    std_times = [result['std_time'] for result in timing_results_Goldin]
    results_Goldin = [result['val'] for result in timing_results_Goldin]
    errors_Goldin = np.abs(np.array(results_True) - np.array(results_Goldin))/np.array(results_True)
    mean_times_Goldin = mean_times.copy()

    line_goldin, = plt.loglog(x_values, mean_times,color='gray', label="Goldin")
    print(f"Goldin mean times: {np.mean(mean_times)} std times: {np.mean(std_times)}")

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
    #plt.ylim(5e-7, 2e-6)
    plt.grid()
    print(num_repeats)
    plt.savefig("Timing_" + str(num_repeats) + ".pdf")
    plt.show()

    #plot errors 
    plt.figure(figsize=(11, 6))
    print(x_values,"\n",errors_Clark)
    plt.loglog(x_values, errors_Clark, label="Clark")
    plt.loglog(x_values, errors_Poly, linestyle="dashed", label="PolyLog")
    plt.loglog(x_values, errors_Rational, linestyle="-.", label="Rational")
    plt.loglog(x_values, errors_Quadrature2,marker="s",  label="Quad n=2")
    plt.loglog(x_values, errors_Quadrature16,marker="o", label="Quad n=16")
    plt.loglog(x_values, errors_Zimmerman,marker="^",  label="Zimmerman")
    plt.loglog(x_values, errors_Goldin, color="gray", label="Goldin")
    # Add emoji markers manually
    for x, y in zip(x_values, errors_Goldin):
        plt.text(x, y, '☭', fontsize=12, ha='center', va='center')  
    """ 
    """  
    plt.xlabel("z")
    plt.ylabel("Relative Error")
    plt.subplots_adjust(right=0.85)  # Adjust the right margin to make room for the legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), handler_map={line_goldin: HandlerLineEmoji()})
    print(num_repeats)
    #plt.savefig("Error.pdf")
    plt.show()

    #plot figure of merit
    plt.figure(figsize=(11, 6))
    plt.loglog(x_values, 1/(mean_times_Clark*errors_Clark), label="Clark")
    plt.loglog(x_values, 1/(mean_times_Poly*errors_Poly), linestyle="dashed", label="PolyLog")
    plt.loglog(x_values, 1/(mean_times_Rational*errors_Rational), linestyle="-.", label="Rational")
    plt.loglog(x_values, 1/(mean_times_Quadrature2*errors_Quadrature2),marker="s",  label="Quad n=2")
    plt.loglog(x_values, 1/(mean_times_Quadrature16*errors_Quadrature16),marker="o", label="Quad n=16")
    plt.loglog(x_values, 1/(mean_times_Zimmerman*errors_Zimmerman),marker="^",  label="Zimmerman")
    plt.loglog(x_values, 1/(mean_times_Goldin*errors_Goldin), color="gray", label="Goldin")
    # Add emoji markers manually
    for x, y in zip(x_values, 1/(mean_times_Goldin*errors_Goldin)):
        plt.text(x,y , '☭', fontsize=12, ha='center', va='center')    
    plt.xlabel("z")
    plt.ylabel("Figure of Merit")
    plt.subplots_adjust(right=0.85)  # Adjust the right margin to make room for the legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), handler_map={line_goldin: HandlerLineEmoji()})
    plt.savefig("FOM_" + str(num_repeats) + ".pdf")
    plt.show()