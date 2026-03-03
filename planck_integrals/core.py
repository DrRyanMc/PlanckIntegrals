"""
Planck and Rosseland Integral Module

This module provides functions to compute incomplete Planck and Rosseland integrals
using rational polynomial approximations. The implementation is based on the 
rational approximation to Π(x) as described in the paper.

Units:
- Frequency (ν): keV (energy units)
- Temperature (T): keV
- Integrated intensity: GJ/(cm²·ns·steradian)

Functions:
- Bg(nu_low, nu_high, T): Incomplete Planck integral over frequency range
- dBgdT(nu_low, nu_high, T): Derivative of Planck integral (Rosseland)
"""

import numpy as np
from numba import njit, prange
from .rational_approx import bRational

# Physical constants in GJ/ns/cm² units
SIGMA_SB = 0.102829  # GJ/(ns·cm²·keV⁴)
SIGMA_SB_OVER_PI = 0.0327314  # GJ/(ns·cm²·keV⁴)


@njit(fastmath=True, cache=True)
def _rosseland_correction(x):
    """
    Compute the Rosseland correction term: -15/(4π⁴) * x⁴/(eˣ - 1)
    From equation (30): Y(x) = Π(x) - 15/(4π⁴) * x⁴/(eˣ - 1)
    
    Parameters:
        x (float): Reduced frequency
        
    Returns:
        float: The correction term value
    """
    if x < 1e-10:  # Avoid division by zero for very small x
        return 0.0
    
    if x > 100:  # Avoid overflow for large x
        return 0.0
    
    PI4 = 97.40909103400243723644  # π⁴
    exp_x = np.exp(x)
    
    return -15.0 / (4.0 * PI4) * (x**4) / (exp_x - 1.0)


@njit(fastmath=True, cache=True)
def Bg(nu_low, nu_high, T):
    """
    Compute the incomplete Planck integral over a frequency range.
    
    This function evaluates the integrated intensity contribution from radiation
    in the frequency range [nu_low, nu_high] at temperature T.
    
    Parameters:
        nu_low (float): Lower frequency bound in keV
        nu_high (float): Upper frequency bound in keV
        T (float): Temperature in keV
        
    Returns:
        float: Integrated intensity in GJ/(cm²·ns·steradian)
        
    Notes:
        - Based on the rational approximation to Π(x)
        - Uses reduced frequency x = ν/T
        - Formula: B = (σ_SB/π) * T^4 * [Π(x_high) - Π(x_low)]
    """
    if T <= 0:
        return 0.0
    
    if nu_high <= nu_low:
        return 0.0
    
    # Convert to reduced frequencies
    x_low = nu_low / T
    x_high = nu_high / T
    
    # Compute Π(x_high) - Π(x_low) using bRational
    pi_diff = bRational(x_low, x_high)
    
    # Multiply by (σ_SB/π) * T^4
    result = SIGMA_SB_OVER_PI * pi_diff * (T**4)
    
    return result


@njit(fastmath=True, cache=True)
def dBgdT(nu_low, nu_high, T):
    """
    Compute the derivative of the Planck integral with respect to temperature.
    This is related to the Rosseland mean and is used in radiation diffusion.
    
    Parameters:
        nu_low (float): Lower frequency bound in keV
        nu_high (float): Upper frequency bound in keV
        T (float): Temperature in keV
        
    Returns:
        float: Temperature derivative in GJ/(cm²·ns·steradian·keV)
        
    Notes:
        - Uses Rosseland function Y(x) from equation (30)
        - Y(x) = Π(x) - 15/(4π⁴) * x⁴/(eˣ - 1)
        - Formula: dB/dT = 4*(σ_SB/π) * T^3 * [Y(x_high) - Y(x_low)]
    """
    if T <= 0:
        return 0.0
    
    if nu_high <= nu_low:
        return 0.0
    
    # Convert to reduced frequencies
    x_low = nu_low / T
    x_high = nu_high / T
    
    # Compute Π(x_high) - Π(x_low) using bRational
    pi_diff = bRational(x_low, x_high)
    
    # Add Rosseland correction terms for Y(x) = Π(x) - 15/(4π⁴) * x⁴/(eˣ - 1)
    # Y(x_high) - Y(x_low) = [Π(x_high) - Π(x_low)] + [correction(x_high) - correction(x_low)]
    rosseland_correction = _rosseland_correction(x_high) - _rosseland_correction(x_low)
    y_diff = pi_diff + rosseland_correction
    
    # Multiply by 4*(σ_SB/π) * T^3
    result = 4.0 * SIGMA_SB_OVER_PI * y_diff * (T**3)
    
    return result


@njit(fastmath=True, cache=True, parallel=True)
def Bg_multigroup(nu_bounds, T):
    """
    Compute Planck integrals for multiple frequency groups in parallel.
    
    Parameters:
        nu_bounds (np.ndarray): Array of frequency group boundaries in keV (length n+1)
        T (float): Temperature in keV
        
    Returns:
        np.ndarray: Array of Planck integrals for each group (length n)
    """
    n = len(nu_bounds) - 1
    result = np.zeros(n)
    
    for i in prange(n):
        result[i] = Bg(nu_bounds[i], nu_bounds[i+1], T)
    
    return result


@njit(fastmath=True, cache=True, parallel=True)
def dBgdT_multigroup(nu_bounds, T):
    """
    Compute Rosseland integrals for multiple frequency groups in parallel.
    
    Parameters:
        nu_bounds (np.ndarray): Array of frequency group boundaries in keV (length n+1)
        T (float): Temperature in keV
        
    Returns:
        np.ndarray: Array of temperature derivatives for each group (length n)
    """
    n = len(nu_bounds) - 1
    result = np.zeros(n)
    
    for i in prange(n):
        result[i] = dBgdT(nu_bounds[i], nu_bounds[i+1], T)
    
    return result
