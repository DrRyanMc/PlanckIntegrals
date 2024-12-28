import numpy as np
import math
from numba import njit

@njit(fastmath=True)
def zimLow(z):
    return 0.0513299112734217*z**3

@njit(fastmath=True)
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

def zimInt(a, b):
    if (b>=1e-3):
        return zimHigh(b) - zimHigh(a)
    else:
        return zimLow(b) - zimLow(a)
if __name__ == "__main__":
    # Example usage
    from MPMathIntegral import compute_Pi
    z_values = [0.9e-3,1.5, 2, 2.5]  # Example inputs
    results = [zimInt(0,z) for z in z_values]

    for z, result in zip(z_values, results):
        print(f"Zimmerman({z}) = {result}")
        result2 = compute_Pi(0,z)
        print(f"MPMath result: {result2}")
        print(f"Error: {abs(result - result2)}")