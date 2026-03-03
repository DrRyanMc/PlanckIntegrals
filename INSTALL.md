# Quick Installation Guide

## For End Users

### Option 1: Install from Source (Recommended)

```bash
cd "PlanckIntegrals"
pip install .
```

Or for editable/development installation:
```bash
pip install -e .
```

### Option 2: Use Standalone Module

If you prefer not to install the package, you can use the standalone files:
- `planck_integral_module.py` (requires `RationalApprox.py` in the same directory)

## Usage After Installation

```python
from planck_integrals import Bg, dBgdT, Bg_multigroup, dBgdT_multigroup

# Compute Planck integral
energy_density = Bg(nu_low=1.0, nu_high=4.0, T=2.0)

# Compute Rosseland integral  
temp_derivative = dBgdT(nu_low=1.0, nu_high=4.0, T=2.0)
```

## Running Examples

```bash
# Run example with installed package
python3 example_installed.py

# Run example with standalone module
python3 planck_integral_module.py
```

## Verification

Quick test:
```bash
python3 -c "from planck_integrals import Bg; print(f'Test: Bg(1.0, 4.0, 2.0) = {Bg(1.0, 4.0, 2.0):.6f} GJ/cm³')"
```

Expected output: `Test: Bg(1.0, 4.0, 2.0) = 0.092094 GJ/cm³`

## Uninstalling

```bash
pip uninstall planck-integrals
```

## For Package Developers

### Building Distribution

```bash
# Install build tools
pip install build twine

# Build wheel and source distribution
python -m build
```

This creates:
- `dist/planck_integrals-1.0.0-py3-none-any.whl` (wheel)
- `dist/planck-integrals-1.0.0.tar.gz` (source)

### Publishing to PyPI

```bash
# Test on TestPyPI first
twine upload --repository testpypi dist/*

# Then publish to PyPI
twine upload dist/*
```

### Testing Installation

```bash
# From TestPyPI
pip install --index-url https://test.pypi.org/simple/ planck-integrals

# From PyPI (after publishing)
pip install planck-integrals
```
