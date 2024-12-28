import numpy as np
from numba import njit

@njit
def PiGoldin(z):
    """
    Evaluate the incomplete Planck integral with Ya. Gol’din and B. N.
    Chetverushkin's approxiamtion. This function uses
    Horner's method for efficient polynomial evaluation.

    Parameters:
        z (float or np.ndarray): Input value(s) for z.

    Returns:
        float or np.ndarray: Evaluated value of σ(z).
    """
    if z <= 2:
        # For z <= 2, use Horner's method for the polynomial
        numerator = (1/3) + z * (-1/8 + z * (1/62.4))
        return z**3 * numerator*0.153989733820265
    else:
        # For z > 2, use Horner's method for the polynomial
        #z**3 + 3*z**2 + 6*z + 7.28
        polynomial = 7.28 + z * (6 + z * (3 + z))
        return 1 - 0.153989733820265*np.exp(-z) * polynomial

def bGoldin(a,b):
    """
    Evaluate the incomplete Planck integral with Ya. Gol’din and B. N.
    Chetverushkin's approxiamtion. This function uses
    Horner's method for efficient polynomial evaluation.

    Parameters:
        a (float): Lower bound of the integral.
        b (float): Upper bound of the integral.

    Returns:
        float: Evaluated value of σ(b) - σ(a).
    """
    return PiGoldin(b) - PiGoldin(a)

if __name__ == "__main__":
    
    # Example usage
    from MPMathIntegral import compute_Pi
        
    z_values = [1.5, 2, 2.5]  # Example inputs
    results = [PiGoldin(z) for z in z_values]

    for z, result in zip(z_values, results):
        print(f"PiGoldin({z}) = {result}")
        result2 = compute_Pi(0,z)
        print(f"MPMath result: {result2}")
        print(f"Error: {abs(result - result2)}")

