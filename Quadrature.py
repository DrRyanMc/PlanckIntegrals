import math
import numpy as np
from numba import njit, prange

@njit(fastmath=True, cache=True, parallel=False)
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
        result += transformed_weights[i] * 0.15398973382026502784*(x_t**3) *eneg/(1-eneg) 
    
    return result
@njit(fastmath=True, cache=True, parallel=True)
def gl_parallel(x,n,N,nodes,weights):
    """
    Compute the Gauss-Legendre quadrature in parallel over n energy groups.

    parameters:
    x (np.ndarray): The energy group bounds
    n (int): number of energy groups (must be length of x - 1)
    N: number of nodes and weights
    nodes (np.ndarray): The nodes for the quadrature.
    weights (np.ndarray): The weights for the quadrature.
    """
    result = np.zeros(n)
    for i in prange(n):
        result[i] = gauss_legendre_quadrature(x[i],x[i+1],N,nodes,weights)
    return result
if __name__ == "__main__":
    # Test gauss_legendre_quadrature
    try:
        from MPMathIntegral import compute_Pi, compute_PiParallel
        a = 0.0
        b = 1.0
        n = 10
        nodes, weights = np.polynomial.legendre.leggauss(n)
        result = gauss_legendre_quadrature(a, b, n, nodes, weights)
        print(f"gauss_legendre_quadrature({a}, {b}, {n}) = {result}")
        result2 = compute_Pi(a,b)
        print(f"Pi({a},{b}) = {result2}")
        print(f"Error = {abs(result - result2)}")

        #test parallel
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = gl_parallel(x, 4, 10, nodes, weights)
        print(f"gauss_legendre_quadrature({x}, 4, 10) = {result}")
        result2 = compute_PiParallel(x,4)
        print(f"MPMath result: {result2}")
        print(f"Max Error: {np.max(np.abs(result - result2))}")
    except Exception as e:
        print(f"Error in gauss_legendre_quadrature: {e}")
        # Perform the quadrature
