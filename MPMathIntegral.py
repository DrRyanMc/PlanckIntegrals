from mpmath import polylog, log, pi
import numpy as np
from math import e 
def compute_Pi_intermediate(x, nofloat=False):
    """
    Compute the value of Pi(x) based on the provided formula using mpmath.

    Parameters:
        x (float): The input value for the formula.

    Returns:
        float: The computed value of Pi(x).
    """
    if (np.abs(x)<1e-12):
        return -1
    # Compute the terms
    term1 = -6 * polylog(4, pow(e, -x))  # 6 * Li_4(e^(-x))
    term2 = -6 *x* polylog(3, pow(e, -x))  # 6 * Li_3(e^(-x))
    term3 = -3 * x**2 * polylog(2, pow(e, -x))  # 3x^2 * Li_2(e^(-x))
    term4 = x**3 * log(1 - pow(e, -x))  # x^3 * log(1 - e^(-x))
    
    # Combine terms with the coefficient
    result = (15 / pi**4) * (term1 + term2 + term3 + term4) #no + 1 because we take difference
    if nofloat:
        return result

    return float(result)  # Convert to float for standard Python compatibility
def compute_Pi(a,b):
    return float(compute_Pi_intermediate(b,nofloat=True) - compute_Pi_intermediate(a,nofloat=True))

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
    except Exception as e:
        print(f"Error in compute_Pi: {e}")