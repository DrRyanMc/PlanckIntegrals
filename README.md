# PlanckIntegrals

Code to numerically evaluate integrals of the Planck blackbody function over a finite range using rational polynomial approximations.

## Overview

This package provides fast, accurate computation of incomplete Planck and Rosseland integrals using rational approximations to the Π(x) function. The implementation is based on the rational approximation method described in the accompanying paper.

## Installation

### Using pip (Recommended)

Install the package from PyPI (once published):

```bash
pip install planck-integrals
```

Or install from source:

```bash
# Clone the repository (or download the source code)
cd PlanckIntegrals

# Install in development mode (editable)
pip install -e .

# Or install normally
pip install .
```

### Dependencies

The package requires:
- `numpy>=1.16.0`
- `numba>=0.50.0`

These will be automatically installed by pip.

### Standalone Usage (Without Installation)

If you prefer not to install the package, you can use the standalone module file:

```python
# Use the standalone file directly (ensure RationalApprox.py is in the same directory)
from planck_integral_module import Bg, dBgdT, Bg_multigroup, dBgdT_multigroup
```

## Module Usage

### Importing the Module

After installation:

```python
from planck_integrals import Bg, dBgdT, Bg_multigroup, dBgdT_multigroup
```

For standalone usage:

```python
from planck_integral_module import Bg, dBgdT, Bg_multigroup, dBgdT_multigroup
```

### Units

- **Frequency (ν)**: keV (energy units)
- **Temperature (T)**: keV
- **Integrated intensity**: GJ/(cm²·ns·steradian)
- **Temperature derivative**: GJ/(cm²·ns·steradian·keV)

### Physical Constants

The module uses the following constants:
- `SIGMA_SB = 0.102829` GJ/(ns·cm²·keV⁴) - Stefan-Boltzmann constant
- `SIGMA_SB_OVER_PI = 0.0327314` GJ/(ns·cm²·keV⁴)

## Functions

### `Bg(nu_low, nu_high, T)`

Compute the incomplete Planck integral over a frequency range.

**Formula**: B = (σ_SB/π) × T⁴ × [Π(x_high) - Π(x_low)]

**Parameters**:
- `nu_low` (float): Lower frequency bound in keV
- `nu_high` (float): Upper frequency bound in keV
- `T` (float): Temperature in keV

**Returns**: Integrated intensity in GJ/(cm²·ns·steradian)

**Example**:
```python
from planck_integral_module import Bg

# Compute Planck integral for a single frequency group
integrated_intensity = Bg(nu_low=1.0, nu_high=4.0, T=2.0)
print(f"Integrated intensity: {integrated_intensity:.6f} GJ/(cm²·ns·steradian)")
```

### `dBgdT(nu_low, nu_high, T)`

Compute the temperature derivative of the Planck integral (Rosseland function).

**Formula**: dB/dT = 4×(σ_SB/π) × T³ × [Y(x_high) - Y(x_low)]

where Y(x) = Π(x) - 15/(4π⁴) × x⁴/(eˣ - 1)

**Parameters**:
- `nu_low` (float): Lower frequency bound in keV
- `nu_high` (float): Upper frequency bound in keV
- `T` (float): Temperature in keV

**Returns**: Temperature derivative in GJ/(cm²·ns·steradian·keV)

**Example**:
```python
from planck_integral_module import dBgdT

# Compute temperature derivative for radiation diffusion
temp_derivative = dBgdT(nu_low=1.0, nu_high=4.0, T=2.0)
print(f"Temperature derivative: {temp_derivative:.6f} GJ/(cm²·ns·steradian·keV)")
```

### `Bg_multigroup(nu_bounds, T)`

Compute Planck integrals for multiple frequency groups in parallel.

**Parameters**:
- `nu_bounds` (np.ndarray): Array of frequency group boundaries in keV (length n+1)
- `T` (float): Temperature in keV

**Returns**: Array of integrated intensities in GJ/(cm²·ns·steradian) for each group (length n)

**Example**:
```python
import numpy as np
from planck_integral_module import Bg_multigroup

# Define energy group structure
group_bounds = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])  # 5 groups
T = 1.5  # keV

# Compute Planck integrals for all groups
integrated_intensities = Bg_multigroup(group_bounds, T)

for i, intensity in enumerate(integrated_intensities):
    print(f"Group {i+1} [{group_bounds[i]:.2f}, {group_bounds[i+1]:.2f}] keV: "
          f"{intensity:.6e} GJ/(cm²·ns·steradian)")
```

### `dBgdT_multigroup(nu_bounds, T)`

Compute Rosseland integrals for multiple frequency groups in parallel.

**Parameters**:
- `nu_bounds` (np.ndarray): Array of frequency group boundaries in keV (length n+1)
- `T` (float): Temperature in keV

**Returns**: Array of temperature derivatives in GJ/(cm²·ns·steradian·keV) for each group (length n)

**Example**:
```python
import numpy as np
from planck_integral_module import dBgdT_multigroup

# Define energy group structure
group_bounds = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
T = 1.5  # keV

# Compute temperature derivatives for all groups
derivatives = dBgdT_multigroup(group_bounds, T)

for i, deriv in enumerate(derivatives):
    print(f"Group {i+1}: dBgdT = {deriv:.6e} GJ/(cm²·ns·steradian·keV)")
```

## Complete Example

```python
import numpy as np
from planck_integral_module import Bg, dBgdT, Bg_multigroup, dBgdT_multigroup

# Example 1: Single frequency group calculation
T = 2.0  # Temperature in keV
nu_low = 1.0  # Lower frequency bound in keV
nu_high = 4.0  # Upper frequency bound in keV

planck_integral = Bg(nu_low, nu_high, T)
rosseland_integral = dBgdT(nu_low, nu_high, T)

print(f"Temperature: {T} keV")
print(f"Frequency range: [{nu_low}, {nu_high}] keV")
print(f"Planck integral: {planck_integral:.6f} GJ/(cm²·ns·steradian)")
print(f"Rosseland integral: {rosseland_integral:.6f} GJ/(cm²·ns·steradian·keV)")

# Example 2: Multigroup calculation
group_bounds = np.logspace(-1, 1, 11)  # 10 groups from 0.1 to 10 keV
T = 1.0  # keV

planck_groups = Bg_multigroup(group_bounds, T)
rosseland_groups = dBgdT_multigroup(group_bounds, T)

print(f"\nMultigroup results at T = {T} keV:")
for i in range(len(planck_groups)):
    print(f"Group {i+1} [{group_bounds[i]:.3f}, {group_bounds[i+1]:.3f}] keV: "
          f"Bg = {planck_groups[i]:.3e} GJ/(cm²·ns·steradian), "
          f"dBgdT = {rosseland_groups[i]:.3e} GJ/(cm²·ns·steradian·keV)")

# Example 3: Computing Planck and Rosseland mean opacities
# Suppose you have absorption coefficients for each group
kappa = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01])  # cm²/g

# Planck mean opacity
planck_mean = np.sum(kappa * planck_groups) / np.sum(planck_groups)

# Rosseland mean opacity
rosseland_mean = np.sum(rosseland_groups) / np.sum(rosseland_groups / kappa)

print(f"\nOpacity-weighted means:")
print(f"Planck mean opacity: {planck_mean:.3f} cm²/g")
print(f"Rosseland mean opacity: {rosseland_mean:.3f} cm²/g")
```

## Performance

The functions are JIT-compiled with Numba for high performance:
- First call will compile the function (slight delay)
- Subsequent calls are very fast
- Multigroup functions use parallel execution for additional speedup

## Accuracy

The rational approximations provide excellent accuracy:
- Full integral (0 to ∞) matches theoretical value to < 0.0001% relative error
- Verified against high-precision Mathematica calculations
- Accurate across all frequency ranges

## Technical Details

- **Planck integral**: Uses the rational approximation to Π(x) where x = ν/T
- **Rosseland integral**: Uses Y(x) = Π(x) - 15/(4π⁴) × x⁴/(eˣ - 1) from equation (30)
- **Implementation**: Three-region piecewise rational approximation for optimal accuracy
- **Dependencies**: Imports `bRational` from `RationalApprox.py` for core calculations

## Testing the Installation

After installing the package, you can verify it's working correctly:

```python
# Quick test
python3 -c "from planck_integrals import Bg; print(f'Bg(1.0, 4.0, 2.0) = {Bg(1.0, 4.0, 2.0):.6f} GJ/(cm²·ns·steradian)')"
```

Expected output: `Bg(1.0, 4.0, 2.0) = 0.092094 GJ/(cm²·ns·steradian)`

Run the example script:
```bash
python3 example_installed.py
```

## Package Structure

```
planck_integrals/
├── __init__.py           # Package initialization and exports
├── core.py               # Main Planck/Rosseland integral functions
└── rational_approx.py    # Rational polynomial approximations

Standalone files (for non-pip usage):
├── planck_integral_module.py  # Standalone module
├── RationalApprox.py          # Core rational approximation
└── example_usage.py           # Usage examples
```

## Development Installation

For development work:

```bash
# Clone the repository
git clone https://github.com/yourusername/planck-integrals.git
cd planck-integrals

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Run tests (if available)
pytest
```

## Publishing to PyPI

To publish the package to PyPI:

```bash
# Install build tools
pip install build twine

# Build the distribution
python -m build

# Upload to PyPI (test first)
twine upload --repository testpypi dist/*

# Upload to production PyPI
twine upload dist/*
```

## Building from Source

To create a standalone distribution:

```bash
# Build wheel and source distribution
python -m build

# The distributions will be in the dist/ directory
# Users can install with: pip install planck_integrals-1.0.0-py3-none-any.whl
```

## Citation

If you use this code in your research, please cite the accompanying paper describing the rational approximation method.
