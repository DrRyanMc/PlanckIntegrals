from PolyLogs import *

#test the Taylor series function
a_value = 0.1  # Example input
max_degree = 31  # Specify the max degree (e.g., a^3)
result = optimized_polynomial_with_degree_numba(a_value, max_degree)
print(f"Value of the polynomial at a={a_value} up to degree {max_degree} is {result}")
