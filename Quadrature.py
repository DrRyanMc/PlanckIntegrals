import math
import numpy as np
from numba import njit

@njit(fastmath=True)
def gauss_legendre_quadrature(a, b, n, nodes, weights):
    """
    Integrate the Planck function using Gauss-Legendre quadrature.

    Parameters:
        a (float): The lower bound of the integral.
        b (float): The upper bound of the integral.
        n (int): The number of nodes and weights.
        nodes (np.ndarray): The nodes for the quadrature.
        weights (np.ndarray): The weights for the quadrature.
    """
    
    # Perform the change of interval
    transformed_nodes = 0.5 * (nodes + 1) * (b - a) + a
    transformed_weights = 0.5 * (b - a) * weights
    
    # Compute the integral approximation
    result = 0.0
    for i in range(n):
        x_t = transformed_nodes[i]
        eneg = math.exp(-x_t)
        #fact = eneg/(1-eneg) * (x_t<=10) + eneg * (1 + eneg + eneg**2/2)*(x_t>10)
        result += transformed_weights[i] * 0.15398973382026502784*(x_t**3) *eneg/(1-eneg) # / (math.exp(x_t) - 1)
    
    return result

if __name__ == "__main__":
    # Test gauss_legendre_quadrature
    try:
        from MPMathIntegral import compute_Pi
        a = 0.0
        b = 1.0
        n = 10
        nodes, weights = np.polynomial.legendre.leggauss(n)
        result = gauss_legendre_quadrature(a, b, n, nodes, weights)
        print(f"gauss_legendre_quadrature({a}, {b}, {n}) = {result}")
        result2 = compute_Pi(a,b)
        print(f"Pi({a},{b}) = {result2}")
        print(f"Error = {abs(result - result2)}")
    except Exception as e:
        print(f"Error in gauss_legendre_quadrature: {e}")
        # Perform the quadrature
