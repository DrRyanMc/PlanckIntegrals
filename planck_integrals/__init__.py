"""
PlanckIntegrals: Fast computation of incomplete Planck and Rosseland integrals

This package provides high-performance functions for computing incomplete Planck
and Rosseland integrals using rational polynomial approximations.

Main functions:
- Bg(nu_low, nu_high, T): Planck integral (energy density)
- dBgdT(nu_low, nu_high, T): Rosseland integral (temperature derivative)
- Bg_multigroup(nu_bounds, T): Multi-group Planck integrals
- dBgdT_multigroup(nu_bounds, T): Multi-group Rosseland integrals

Units:
- Frequency (ν): keV
- Temperature (T): keV
- Energy density: GJ/cm³
- Temperature derivative: GJ/(cm³·keV)

Example:
    >>> from planck_integrals import Bg, dBgdT
    >>> energy_density = Bg(nu_low=1.0, nu_high=4.0, T=2.0)
    >>> temp_derivative = dBgdT(nu_low=1.0, nu_high=4.0, T=2.0)
"""

from .core import (
    Bg,
    dBgdT,
    Bg_multigroup,
    dBgdT_multigroup,
    SIGMA_SB,
    SIGMA_SB_OVER_PI
)

from .rational_approx import (
    bRational,
    bRationalParallel
)

__version__ = "1.0.0"
__author__ = "Ryan McClarren"

__all__ = [
    'Bg',
    'dBgdT',
    'Bg_multigroup',
    'dBgdT_multigroup',
    'bRational',
    'bRationalParallel',
    'SIGMA_SB',
    'SIGMA_SB_OVER_PI',
]
