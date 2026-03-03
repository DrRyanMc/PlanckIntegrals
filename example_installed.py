"""
Example usage of the planck_integrals package

This script demonstrates how to use the installed package
for computing Planck and Rosseland integrals.
"""

import numpy as np
from planck_integrals import Bg, dBgdT, Bg_multigroup, dBgdT_multigroup

def main():
    print("=" * 70)
    print("Planck Integrals Package - Example Usage")
    print("=" * 70)
    
    # Example 1: Single frequency group
    print("\n1. Single Frequency Group Calculation")
    print("-" * 70)
    T = 2.0  # Temperature in keV
    nu_low = 1.0  # Lower frequency bound in keV
    nu_high = 4.0  # Upper frequency bound in keV
    
    planck = Bg(nu_low, nu_high, T)
    rosseland = dBgdT(nu_low, nu_high, T)
    
    print(f"Temperature: {T} keV")
    print(f"Frequency range: [{nu_low}, {nu_high}] keV")
    print(f"Planck integral:    Bg = {planck:.6f} GJ/cm³")
    print(f"Rosseland integral: dBgdT = {rosseland:.6f} GJ/(cm³·keV)")
    
    # Example 2: Temperature sweep
    print("\n2. Temperature Sweep")
    print("-" * 70)
    temperatures = np.array([0.5, 1.0, 2.0, 5.0])
    nu_low, nu_high = 0.1, 2.0
    
    print(f"Frequency range: [{nu_low}, {nu_high}] keV")
    print(f"{'T (keV)':>10s} {'Bg (GJ/cm³)':>15s} {'dBgdT (GJ/(cm³·keV))':>25s}")
    for T in temperatures:
        planck = Bg(nu_low, nu_high, T)
        rosseland = dBgdT(nu_low, nu_high, T)
        print(f"{T:10.2f} {planck:15.6e} {rosseland:25.6e}")
    
    # Example 3: Multigroup calculation
    print("\n3. Multigroup Calculation")
    print("-" * 70)
    nu_bounds = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    T = 1.5  # keV
    
    planck_groups = Bg_multigroup(nu_bounds, T)
    rosseland_groups = dBgdT_multigroup(nu_bounds, T)
    
    print(f"Temperature: {T} keV")
    print(f"Energy group structure: {len(planck_groups)} groups")
    print(f"\n{'Group':>5s} {'ν_low (keV)':>12s} {'ν_high (keV)':>13s} "
          f"{'Bg (GJ/cm³)':>15s} {'dBgdT (GJ/(cm³·keV))':>25s}")
    
    for i in range(len(planck_groups)):
        print(f"{i+1:5d} {nu_bounds[i]:12.2f} {nu_bounds[i+1]:13.2f} "
              f"{planck_groups[i]:15.6e} {rosseland_groups[i]:25.6e}")
    
    print(f"\nTotal energy density: {np.sum(planck_groups):.6e} GJ/cm³")
    
    # Example 4: Computing opacity-weighted means
    print("\n4. Planck and Rosseland Mean Opacities")
    print("-" * 70)
    
    # Example absorption coefficients (cm²/g) for each group
    kappa = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
    print(f"Example absorption coefficients: {kappa} cm²/g")
    
    # Planck mean opacity
    planck_mean = np.sum(kappa * planck_groups) / np.sum(planck_groups)
    
    # Rosseland mean opacity
    rosseland_mean = np.sum(rosseland_groups) / np.sum(rosseland_groups / kappa)
    
    print(f"\nPlanck mean opacity:    κ_P = {planck_mean:.4f} cm²/g")
    print(f"Rosseland mean opacity: κ_R = {rosseland_mean:.4f} cm²/g")
    
    # Example 5: Full integral verification
    print("\n5. Full Integral Verification")
    print("-" * 70)
    from planck_integrals import SIGMA_SB_OVER_PI
    
    T = 1.0
    full_integral = Bg(0.0, 100.0, T)  # Effectively 0 to infinity
    theoretical = SIGMA_SB_OVER_PI * T**4
    
    print(f"Temperature: {T} keV")
    print(f"Numerical integral (0 to 100 keV): {full_integral:.8f} GJ/cm³")
    print(f"Theoretical value (σ_SB/π)T⁴:      {theoretical:.8f} GJ/cm³")
    print(f"Relative error: {abs(full_integral - theoretical)/theoretical * 100:.4f}%")
    
    print("\n" + "=" * 70)
    print("All calculations completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
