import numpy as np
from numba import njit, jit
import matplotlib.pyplot as plt
import math

from numba import njit
import numpy as np

@njit(fastmath=True)
def optimized_polynomial_with_degree_numba(a, max_degree=21):
    """
    Compute the polynomial up to the specified max degree efficiently using Horner's method with Numba.
    This is the Taylor polynomial for the integral of the Planck function
    from 0 to a

    Parameters:
        a (float): Upper bound of integral
        max_degree (int): Maximum degree of the polynomial (up to 31).
    
    Returns:
        float: Value of the polynomial for the given input.
    """
    if max_degree > 31 or max_degree < 0:
        raise ValueError("max_degree must be between 0 and 31")
    
    # Coefficients for terms from a^31 down to a^3
    coefficients_full = np.array([
        -3392780147 / 1174691236311131831103651840000000,  # a^31
        0,  # a^30
        657931 / 5397901095079183122432000000,  # a^29
        0,  # a^28
        -236364091 / 45733251691757079075225600000,  # a^27
        0,  # a^26
        77683 / 352527500984795136000000,  # a^25
        0,  # a^24
        -174611 / 18465726242060697600000,  # a^23
        0,  # a^22
        43867 / 107290978560589824000,  # a^21
        0,  # a^20
        -3617 / 202741834014720000,  # a^19
        0,  # a^18
        1 / 1270312243200,  # a^17
        0,  # a^16
        -691 / 19615115520000,  # a^15
        0,  # a^14
        1 / 622702080,  # a^13
        0,  # a^12
        -1 / 13305600,  # a^11
        0,  # a^10
        1 / 272160,  # a^9
        0,  # a^8
        -1 / 5040,  # a^7
        0,  # a^6
        1 / 60,  # a^5
        -1 / 8,  # a^4
        1 / 3,  # a^3
        0,  # a^2
        0,  # a^1
        0,  # a^0
    ], dtype=np.float64)
    
    # Filter coefficients corresponding to terms up to max_degree
    coefficients = coefficients_full[31 - max_degree:]
    
    # Start Horner's method
    result = 0.0
    for coeff in coefficients:
        result = result * a + coeff  # Iteratively build the polynomial
    
    return (15 / np.pi**4)*result

@njit(fastmath=True)
def compute_Pi_L(x, L):
    """
    Use the L term power series representation of the Polylogarithm
    to integrate the Planck function from 0 to a.

    Parameters:
        x (float): The input variable.
        L (int): The upper limit of the summation.

    Returns:
        float: The computed value of Pi_L(x).
    """
    # Precompute exponential terms
    l_values = np.arange(1, L + 1)  # l from 1 to L
    exp_terms = np.exp(-l_values * x)  # e^(-lx)
    
    # Compute the summation terms
    sum_1 = np.sum(exp_terms / l_values)
    sum_2 = np.sum(exp_terms / l_values**2)
    sum_3 = np.sum(exp_terms / l_values**3)
    sum_4 = np.sum(exp_terms / l_values**4)
    
    # Combine terms based on the equation
    Pi_L = 1 + (15 / np.pi**4) * (
        -x**3 * sum_1
        - 3 * x**2 * sum_2
        - 6 * x * sum_3
        - 6 * sum_4
    )
    
    return Pi_L