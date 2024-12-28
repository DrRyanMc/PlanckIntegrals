import numpy as np
import pytest
from PolyLogs import optimized_polynomial_with_degree_numba, compute_Pi_L

# FILE: test_PolyLogs.py


def test_optimized_polynomial_with_degree_numba():
    # Test with known values
    assert np.isclose(optimized_polynomial_with_degree_numba(1.0, 3), 0.3183098861837907, atol=1e-8)
    assert np.isclose(optimized_polynomial_with_degree_numba(0.5, 5), 0.07957747154594767, atol=1e-8)
    
    # Test with edge cases
    with pytest.raises(ValueError):
        optimized_polynomial_with_degree_numba(1.0, 32)
    with pytest.raises(ValueError):
        optimized_polynomial_with_degree_numba(1.0, -1)

def test_compute_Pi_L():
    # Test with known values
    assert np.isclose(compute_Pi_L(1.0, 10), 0.9999999999999999, atol=1e-8)
    assert np.isclose(compute_Pi_L(0.5, 10), 1.0000000000000002, atol=1e-8)
    
    # Test with edge cases
    print(compute_Pi_L(0.0, 10))
    assert np.isclose(compute_Pi_L(0.0, 10), 1.0, atol=1e-8)
    assert np.isclose(compute_Pi_L(1.0, 1), 0.36787944117144233, atol=1e-8)

if __name__ == "__main__":
    pytest.main()