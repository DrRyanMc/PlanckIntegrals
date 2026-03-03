"""
Example of using the planck_integral_module in another program
"""

import numpy as np
from planck_integral_module import Bg, dBgdT, Bg_multigroup, dBgdT_multigroup

# Example 1: Simple single group calculation
print("Example 1: Single group")
print("-" * 40)
T = 2  # Temperature in keV
nu_low = 1  # Lower bound in keV
nu_high = 4  # Upper bound in keV

planck_integral = Bg(nu_low, nu_high, T)
rosseland_integral = dBgdT(nu_low, nu_high, T)

print(f"Temperature: {T} keV")
print(f"Frequency range: [{nu_low}, {nu_high}] keV")
print(f"Planck integral Bg = {planck_integral:.6e}")
print(f"Rosseland integral dBgdT = {rosseland_integral:.6e}")

# Example 2: Multigroup radiation transport
print("\n\nExample 2: Multigroup structure")
print("-" * 40)

# Define energy groups (in keV)
group_bounds = np.logspace(-2, 1, 11)  # 10 groups from 0.01 to 10 keV
T = 1.0  # keV

planck_groups = Bg_multigroup(group_bounds, T)
rosseland_groups = dBgdT_multigroup(group_bounds, T)

print(f"Temperature: {T} keV")
print(f"Number of groups: {len(group_bounds)-1}")
print("\nGroup    Energy Range (keV)        Bg           dBgdT")
print("-" * 65)
for i in range(len(planck_groups)):
    print(f"{i+1:3d}    [{group_bounds[i]:7.4f}, {group_bounds[i+1]:7.4f}]   "
          f"{planck_groups[i]:11.5e}  {rosseland_groups[i]:11.5e}")

# Example 3: Temperature sweep
print("\n\nExample 3: Temperature dependence")
print("-" * 40)

nu_low, nu_high = 0.1, 10.0
temperatures = np.linspace(0.5, 5.0, 10)

print(f"Frequency range: [{nu_low}, {nu_high}] keV")
print("\nT (keV)      Bg            dBgdT")
print("-" * 40)
for T in temperatures:
    bg = Bg(nu_low, nu_high, T)
    dbdt = dBgdT(nu_low, nu_high, T)
    print(f"{T:5.2f}    {bg:12.6e}  {dbdt:12.6e}")

# Example 4: Computing opacity-weighted integrals (conceptual)
print("\n\nExample 4: With absorption coefficients")
print("-" * 40)

# Suppose you have absorption coefficients for each group
group_bounds = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
kappa = np.array([10.0, 5.0, 2.0, 1.0, 0.5])  # Example opacities in cm^2/g

T = 1.5  # keV
planck_groups = Bg_multigroup(group_bounds, T)
rosseland_groups = dBgdT_multigroup(group_bounds, T)

# Compute Planck and Rosseland means (simplified)
planck_mean_opacity = np.sum(kappa * planck_groups) / np.sum(planck_groups)
rosseland_mean_opacity = np.sum(rosseland_groups) / np.sum(rosseland_groups / kappa)

print(f"Temperature: {T} keV")
print(f"Planck mean opacity: {planck_mean_opacity:.3f} cm^2/g")
print(f"Rosseland mean opacity: {rosseland_mean_opacity:.3f} cm^2/g")
