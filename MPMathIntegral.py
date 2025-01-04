from mpmath import polylog, log, pi
import numpy as np
import math
from math import e 
def compute_Pi_intermediate(x, nofloat=False):
    """
    Compute the value of Pi(x) based on the provided formula using mpmath.

    Parameters:
        x (float): The input value for the formula.

    Returns:
        float: The computed value of Pi(x).
    """
    if (math.fabs(x)<1e-12):
        return -1
    # Compute the terms
    term1 = -6 * polylog(4, pow(e, -x))  # 6 * Li_4(e^(-x))
    term2 = -6 *x* polylog(3, pow(e, -x))  # 6 * Li_3(e^(-x))
    term3 = -3 * x**2 * polylog(2, pow(e, -x))  # 3x^2 * Li_2(e^(-x))
    term4 = x**3 * log(1 - pow(e, -x))  # x^3 * log(1 - e^(-x))
    
    # Combine terms with the coefficient
    result = 0.153989733820265027837291749007 * (term1 + term2 + term3 + term4) #no + 1 because we take difference
    if nofloat:
        return result

    return float(result)  # Convert to float for standard Python compatibility
def compute_Pi(a,b):
    return float(compute_Pi_intermediate(b,nofloat=True) - compute_Pi_intermediate(a,nofloat=True))
def compute_PiParallel(x,n):
    """
    Compute the value of Pi(x) based on the provided formula using mpmath.

    Parameters:
        x (np.ndarray): The energy group bounds
        n (int): number of energy groups (must be length of x - 1)

    Returns:
        np.ndarray: The computed values of Pi(x).
    """
    result = np.zeros(n)
    for i in range(n):
        result[i] = compute_Pi(x[i],x[i+1])
    return result
    
    return result
if __name__ == "__main__":
    # Test compute_Pi
    try:
        result = compute_Pi(0, 1.0)
        print(f"compute_Pi(0, 1.0) = {result}")
        mathematica_result = 0.034617691065528858418
        print(f"Mathematica gives = {mathematica_result}")
        print(f"Error = {abs(result - mathematica_result)}")

        # Example Usage
        x_value = 10  # Example input
        result = compute_Pi(19.9,50)
        print(f"Pi({x_value}) = {result}")
        mathematica_result = 3.2250573960478671454e-6
        print(f"Mathematica gives = {mathematica_result}")
        print(f"Error = {abs(result - mathematica_result)}")

        import matplotlib.pyplot as plt
        import csv
        filename = "./BenchmarkResults/DPiDz.csv"
        # Load the third column into a NumPy array
        tmp = np.loadtxt(filename, delimiter=',', skiprows=1)
        plt.figure(figsize=(8, 5))
        x_values = np.linspace(1e-12, 35, 1000)
        #plt.plot(x_values, [-np.log(1- x) for x in x_values], label='Li$_1$(x)', linewidth=2)
        #plt.plot(x_values, [compute_Pi(0,x) for x in x_values], linewidth=2, label='$\Pi(x)$')
        plt.plot(x_values, [compute_Pi(0,x)/x for x in x_values], linewidth=2, label='$\\frac{\Pi(x)}{x}$')
        
        tmp[-2,:] = (30,-0.00111111)
        tmp[-1,:] = (35,-0.000816327)
        plt.plot(tmp[:,0], tmp[:,1], '--', label='$\\frac{d}{dx}\left(\\frac{\Pi(x)}{x}\\right)$', linewidth=2)
        plt.plot([4.60834547959553635713012610745557043569],
                 [compute_Pi(0,4.60834547959553635713012610745557043569)/4.60834547959553635713012610745557043569], "o",
                 markerfacecolor="none", markersize=10, markeredgewidth=1, 
                 markeredgecolor="black", label='Maximum')
        #plt.title('Dilogarithm Function (Li₂(x)) from 0 to 1')
        plt.xlabel('x')
        plt.ylabel('Function Value')
        plt.grid(True, linestyle="--")
        plt.legend(fontsize=12)
        ax = plt.gca()  # Get the current axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig("Pi_func.pdf")
        plt.show()

    except Exception as e:
        print(f"Error in compute_Pi: {e}")

        