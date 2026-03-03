"""
Planck and Rosseland Integral Module

This module provides functions to compute incomplete Planck and Rosseland integrals
using rational polynomial approximations. The implementation is based on the 
rational approximation to Π(x) as described in the paper.

Units:
- Frequency (ν): keV (energy units)
- Temperature (T): keV
- Energy density: GJ (gigajoules)

Functions:
- Bg(nu_low, nu_high, T): Incomplete Planck integral over frequency range
- dBgdT(nu_low, nu_high, T): Derivative of Planck integral (Rosseland)

Author: Generated from RationalApprox.py
"""

import numpy as np
from numba import njit, prange
from RationalApprox import bRational

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
    
    This function evaluates the energy density contribution from radiation
    in the frequency range [nu_low, nu_high] at temperature T.
    
    Parameters:
        nu_low (float): Lower frequency bound in keV
        nu_high (float): Upper frequency bound in keV
        T (float): Temperature in keV
        
    Returns:
        float: Energy density in GJ/cm³
        
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
        float: Temperature derivative in GJ/(cm³·keV)
        
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


# Non-JIT versions for convenience
def Bg_array(nu_low, nu_high, T):
    """
    Vectorized version of Bg that handles arrays.
    
    Parameters:
        nu_low: Lower frequency bound(s) in keV (scalar or array)
        nu_high: Upper frequency bound(s) in keV (scalar or array)
        T: Temperature(s) in keV (scalar or array)
        
    Returns:
        Planck integral value(s)
    """
    nu_low = np.atleast_1d(nu_low)
    nu_high = np.atleast_1d(nu_high)
    T = np.atleast_1d(T)
    
    result = np.zeros(max(len(nu_low), len(nu_high), len(T)))
    
    for i in range(len(result)):
        nl = nu_low[i] if len(nu_low) > 1 else nu_low[0]
        nh = nu_high[i] if len(nu_high) > 1 else nu_high[0]
        t = T[i] if len(T) > 1 else T[0]
        result[i] = Bg(nl, nh, t)
    
    return result


def dBgdT_array(nu_low, nu_high, T):
    """
    Vectorized version of dBgdT that handles arrays.
    
    Parameters:
        nu_low: Lower frequency bound(s) in keV (scalar or array)
        nu_high: Upper frequency bound(s) in keV (scalar or array)
        T: Temperature(s) in keV (scalar or array)
        
    Returns:
        Temperature derivative value(s)
    """
    nu_low = np.atleast_1d(nu_low)
    nu_high = np.atleast_1d(nu_high)
    T = np.atleast_1d(T)
    
    result = np.zeros(max(len(nu_low), len(nu_high), len(T)))
    
    for i in range(len(result)):
        nl = nu_low[i] if len(nu_low) > 1 else nu_low[0]
        nh = nu_high[i] if len(nu_high) > 1 else nu_high[0]
        t = T[i] if len(T) > 1 else T[0]
        result[i] = dBgdT(nl, nh, t)
    
    return result





@njit(fastmath=True, cache=True, parallel=True)
def Bg_multigroup_GJ(nu_bounds, T):
    """
    Compute Planck integrals for multiple frequency groups with output in GJ/cm^3.
    
    Note: This is now an alias for Bg_multigroup() since that function already
    returns values in GJ/cm^3 with proper physical constants.
    
    Parameters:
        nu_bounds (np.ndarray): Array of frequency group boundaries in keV (length n+1)
        T (float): Temperature in keV
        
    Returns:
        np.ndarray: Array of energy densities in GJ/cm^3 for each group (length n)
    """
    return Bg_multigroup(nu_bounds, T)


@njit(fastmath=True, cache=True, parallel=True)
def dBgdT_multigroup_GJ(nu_bounds, T):
    """
    Compute temperature derivatives for multiple frequency groups with output in GJ/(cm^3·keV).
    
    Note: This is now an alias for dBgdT_multigroup() since that function already
    returns values in GJ/(cm^3·keV) with proper physical constants.
    
    Parameters:
        nu_bounds (np.ndarray): Array of frequency group boundaries in keV (length n+1)
        T (float): Temperature in keV
        
    Returns:
        np.ndarray: Array of temperature derivatives in GJ/(cm^3·keV) for each group (length n)
    """
    return dBgdT_multigroup(nu_bounds, T)


if __name__ == "__main__":
    # Example usage and basic tests
    print("Planck Integral Module - Example Usage")
    print("=" * 50)
    
    # Test 1: Single frequency group
    T = 1.0  # 1 keV temperature
    nu_low = 0.1  # 0.1 keV
    nu_high = 2.0  # 2 keV
    
    planck = Bg(nu_low, nu_high, T)
    rosseland = dBgdT(nu_low, nu_high, T)
    
    
    print(f"\nTest 1: Single frequency group")
    print(f"  Temperature: {T} keV")
    print(f"  Frequency range: [{nu_low}, {nu_high}] keV")
    print(f"  Bg(nu_low, nu_high, T) = {planck:.6e} GJ/cm³")
    print(f"  dBgdT(nu_low, nu_high, T) = {rosseland:.6e} GJ/(cm³·keV)")
    
    # Test 2: Multiple temperature values
    temperatures = np.array([0.5, 1.0, 2.0, 5.0])
    print(f"\nTest 2: Multiple temperatures, fixed frequency range [{nu_low}, {nu_high}] keV")
    for temp in temperatures:
        planck = Bg(nu_low, nu_high, temp)
        rosseland = dBgdT(nu_low, nu_high, temp)
        print(f"  T = {temp:4.1f} keV: Bg = {planck:12.6e} GJ/cm³, dBgdT = {rosseland:12.6e} GJ/(cm³·keV)")
    
    # Test 3: Multigroup calculation
    nu_bounds = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    T = 1.0
    
    planck_groups = Bg_multigroup(nu_bounds, T)
    rosseland_groups = dBgdT_multigroup(nu_bounds, T)
    
    print(f"\nTest 3: Multigroup calculation at T = {T} keV")
    print(f"  Group boundaries: {nu_bounds} keV")
    print(f"  Planck integrals by group:")
    for i, (bg_val, dbdt_val) in enumerate(zip(planck_groups, rosseland_groups)):
        print(f"    Group {i+1} [{nu_bounds[i]:.2f}, {nu_bounds[i+1]:.2f}]: "
              f"Bg = {bg_val:10.6e} GJ/cm³, dBgdT = {dbdt_val:10.6e} GJ/(cm³·keV)")
    
    # Test 4: Check normalization (integrate over all frequencies)
    nu_bounds_full = np.array([0.0, 100.0])  # Effectively 0 to infinity in reduced units
    T = 1.0
    full_integral = Bg(nu_bounds_full[0], nu_bounds_full[1], T)
    print(f"\nTest 4: Full integral (0 to 100 keV at T=1 keV)") 
    print(f"  Bg(0, 100, 1.0) = {full_integral:.8f} GJ/cm³")
    print(f"  Should equal (\u03c3_SB/\u03c0)*T^4 = {SIGMA_SB_OVER_PI:.8f} GJ/cm³")
    print(f"  Relative error: {abs(full_integral - SIGMA_SB_OVER_PI)/SIGMA_SB_OVER_PI * 100:.4f}%")
    
    # Test 5: Verification against Mathematica
    nu_low_test = 1.0  # keV
    nu_high_test = 4.0  # keV
    T_test = 2.0  # keV
    

    upper_part = bRational(0,nu_high/T)
    upper_mathematica =0.181145

    result_GJ_cm3 = Bg(nu_low_test, nu_high_test, T_test)
    expected_mathematica = 0.0920939  # GJ/cm^3

    rosseland_result = dBgdT(nu_low_test, nu_high_test, T_test)
    expected_mathematica_rosseland = 0.0870939
    print(f"\nTest 5: Verification against Mathematica")
    print(f"  Parameters: nu_low={nu_low_test} keV, nu_high={nu_high_test} keV, T={T_test} keV")
    print(f"  Bg result: {result_GJ_cm3:.6f} GJ/cm³")
    print(f"  Expected (Mathematica): {expected_mathematica:.6f} GJ/cm^3")
    print(f"  Relative error: {abs(result_GJ_cm3 - expected_mathematica)/expected_mathematica * 100:.4f}%")
    print(f"  Upper part: {upper_part:.6f} GJ/cm³, expected from Mathematica {upper_mathematica}")
    print(f"  dBgdT result: {rosseland_result:.6f} GJ/(cm³·keV)")
    print(f"  Expected (Mathematica): {expected_mathematica_rosseland:.6f} GJ/(cm³·keV)")
    print(f"  Relative error: {abs(rosseland_result - expected_mathematica_rosseland)/expected_mathematica_rosseland * 100:.4f}%")
    
    
    
