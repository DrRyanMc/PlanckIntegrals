import numpy as np
import math
from numba import njit, prange

@njit(fastmath=True, cache=True)
def zimLow(z):
    return 0.0513299112734217*z**3

@njit(fastmath=True, cache=True)
def zimHigh(z):
    # Constants for the polynomial
    a = [0.07713864107538, 0.5194172986679, 2.161761553097,
          5.570970415031,8.317008834543,  6.493939402267]  

    b = [0.07713864107538,.2807339758744,1.0]  # Denominator coefficients

    numerator = 0.0
    for coeff in (a):
        numerator = numerator * z + coeff
    denominator = 0.0
    for coeff in (b):
        denominator = denominator * z + coeff
    
    return 1 - 0.153989733820265 * numerator/denominator * math.exp(-z)
@njit(fastmath=True, cache=True)
def zimInt(a, b):
    if (b>=1e-3):
        return zimHigh(b) - zimHigh(a)
    else:
        return zimLow(b) - zimLow(a)
@njit(fastmath=True, cache=True, parallel=True)
def zimIntParallel(x,n):
    """
    Evalaute the Planck Integral using the rational polynomial approximation in parallel over a n
    number of groups

    parameters:
    x (np.ndarray): The energy group bounds
    n (int): number of energy groups (must be length of x - 1)
    """
    result = np.zeros(n)
    for i in prange(n):
        result[i] = zimInt(x[i],x[i+1])
    return result
if __name__ == "__main__":
    # Example usage
    from MPMathIntegral import compute_Pi, compute_PiParallel
    z_values = [0.9e-3,1.5, 2, 2.5]  # Example inputs
    results = [zimInt(0,z) for z in z_values]

    for z, result in zip(z_values, results):
        print(f"Zimmerman({z}) = {result}")
        result2 = compute_Pi(0,z)
        print(f"MPMath result: {result2}")
        print(f"Error: {abs(result - result2)}")
    #test parallel
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    result = zimIntParallel(x, 4)
    print(f"zimIntParallel({x}, 4) = {result}")
    result2 = compute_PiParallel(x,4)
    print(f"MPMath result: {result2}")
    print(f"Max Error: {np.max(np.abs(result - result2))}")
    