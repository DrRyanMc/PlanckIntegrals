import numpy as np
import math
from numba import njit,prange

@njit(fastmath=True, cache=True)
def PlanckIntA(z):
    """
    Evaluate the polynomial approximation to the Planck Integral using Horner's method when
    z <= 1.

    Parameters:
        z (float): The input value for the polynomial.

    Returns:
        float: The evaluated value of the polynomial.
    """
    # Coefficients of the polynomial, from highest to lowest degree
    coefficients = [
        1.08750502639625e-9,
        -1.37466236529825e-8,
        2.48265186428919e-9,
        5.64058261139052e-7,
        7.73024978191323e-10,
        -0.0000305537299528005,
        3.38532276078157e-11,
        0.00256649556081953,
        -0.0192487167274314,
        0.0513299112734207,
        0, #0*8.95033840578918e-19,
        0, #0*-5.80299105212354e-23,
        0 #0*5.80298210178641e-33
    ]

    # Horner's method: Evaluate the polynomial
    result = 0.0
    for coeff in coefficients:
        result = result * z + coeff

    return result

@njit(fastmath=True, cache=True)
def PlanckIntB(z):
    """
    Evaluate the rational polynomial approximation to Planck's Integral
    efficiently using Horner's method when z is in the range [1, 4.60834547959553635713012610745557043569].

    Parameters:
        z (float): The input value for the rational polynomial. 
        Should be in [1,4.60834547959553635713012610745557043569]

    Returns:
        float: The evaluated value of the rational polynomial.
    """
    # Coefficients of the numerator (highest to lowest degree)
    # Coefficients for the numerator (highest degree first)
    numerator_coeffs = [
        1.37839794361116e-13,
        -4.41475932122409e-12,
        2.16011320042569e-10,
        -3.32386983347469e-9,
        6.77941964950341e-8,
        -4.0110799985171e-7,
        5.55171714310303e-6,
        1.61066450038861e-6,
        0.000184583709948367,
        0.0017627896470925,
        0.00243407740331316,
        0.0513299112734217,
        -1.50803009236045e-17,
        2.10908974464015e-18,
        -1.41249404058269e-19
    ]

    # Coefficients for the denominator (highest degree first)
    denominator_coeffs = [
        8.4051245916206e-16,
        3.60098845901839e-14,
        1.21211579958412e-12,
        3.1014248231405e-11,
        6.36909463124141e-10,
        1.08080345539615e-8,
        1.53974339199716e-7,
        1.84136729899659e-6,
        0.000018842424092001,
        0.000159466338512454,
        0.00118108372266407,
        0.00699146040951256,
        0.036006242042683,
        0.142749942493712,
        0.422420253472632,
        1.0
    ]

    # Horner's method for the numerator
    numerator = 0.0
    for coeff in numerator_coeffs:
        numerator = numerator * z + coeff

    # Horner's method for the denominator
    denominator = 0.0
    for coeff in denominator_coeffs:
        denominator = denominator * z + coeff

    # Compute the rational polynomial
    return numerator / denominator

@njit(fastmath=True, cache=True)
def PlanckIntC(z):
    """
    Evaluate the rational polynomial approximation to Planck's Integral using Horner's method.

    Parameters:
        z (float): The input value for the polynomial.
        Should be in the range [4.60834547959553635713012610745557043569, inf].

    Returns:
        float: The evaluated value of the rational polynomial.
    """
    # Coefficients for the numerator (highest to lowest degree)
    numerator_coefficients = [
        2.60918226968872e-23,
        -1.5717908318742e-20,
        4.41827357257837e-18,
        -7.67993976921638e-16,
        9.21902324802904e-14,
        -8.07975976849841e-12,
        5.32631745507925e-10,
        -2.67840332691655e-8,
        1.02939262166793e-6,
        -0.0000299266424885916,
        0.00064067417751766,
        -0.00953770584997182,
        0.0858480323226633,
        -0.239927670638369,
        -3.15998509049712,
        25.6572909391779,
    ]

    # Coefficients for the denominator (highest to lowest degree)
    denominator_coefficients = [
        8.11756393563567e-19,
        -1.20137760129307e-16,
        9.17011702895401e-15,
        -4.67744488675722e-13,
        1.74053936269384e-11,
        -5.01961354412149e-10,
        1.14549652872828e-8,
        -2.13098148369549e-7,
        3.2289443728847e-6,
        -0.0000408696607789196,
        0.000422901460469686,
        -0.00372727464164911,
        0.0261495465830976,
        -0.162986793115613,
        0.725820694406949,
        -3.34829549552425,
        7.26162846427674,
        -28.5898229933688,
        1.0,
    ]

    # Evaluate numerator using Horner's method
    numerator = 0.0
    #for coeff in numerator_coefficients:
    #    numerator = numerator * z + coeff
    for i in prange(len(numerator_coefficients)):
        numerator = numerator * z + numerator_coefficients[i]
    # Evaluate denominator using Horner's method
    denominator = 0.0
    for coeff in denominator_coefficients:
        denominator = denominator * z + coeff

    # Return the rational polynomial value

    return z*numerator / denominator

@njit(fastmath=True, cache=True)
def bRational(a,b):
    """
    Evaluate the rational polynomial approximation to the Planck Integral using Horner's method.
    """
    if (a < 1) and (b < 1):
        return PlanckIntA(b) - PlanckIntA(a)
    elif (a < 1) and (b < 4.60834547959553635713012610745557043569):
        return PlanckIntB(b) - PlanckIntA(a)
    elif (a < 1) and (b >= 4.60834547959553635713012610745557043569):
        return (PlanckIntC(b) - PlanckIntA(a)) + 1. 
    elif (a < 4.60834547959553635713012610745557043569) and (b < 4.60834547959553635713012610745557043569):
        return PlanckIntB(b) - PlanckIntB(a)
    elif (a < 4.60834547959553635713012610745557043569) and (b >= 4.60834547959553635713012610745557043569):
        return PlanckIntC(b) - PlanckIntB(a) + 1
    elif (a >= 4.60834547959553635713012610745557043569) and (b >= 4.60834547959553635713012610745557043569):
        return PlanckIntC(b) - PlanckIntC(a)
    else:
        assert 0, "Invalid input values"
@njit(fastmath=True, cache=True, parallel=True)
def bRationalParallel(x,n):
    """
    Evalaute the Planck Integral using the rational polynomial approximation in parallel over a n
    number of groups

    parameters:
    x (np.ndarray): The energy group bounds
    n (int): number of energy groups (must be length of x - 1)
    """
    result = np.zeros(n)
    for i in prange(len(x)-1):
        result[i] = bRational(x[i],x[i+1])
    return result
if __name__ == "__main__":
    # Test the polynomial evaluation
    try:
        from MPMathIntegral import compute_Pi, compute_PiParallel
        # Example usage
        z_value = 0.5  # Replace with the desired value of z
        result = PlanckIntA(z_value)
        print(f"Polynomial evaluated at z = {z_value}: {result}")
        result2 = compute_Pi(0, z_value)
        print(f"MPMath result: {result2}")
        print(f"Error: {abs(result - result2)}")



        # Example usage
        z_value = 1.5  # Replace with the desired value of z
        result = PlanckIntB(z_value)
        print(f"Rational polynomial evaluated at z = {z_value}: {result}")
        result2 = compute_Pi(0, z_value)
        print(f"MPMath result: {result2}")
        print(f"Error: {abs(result - result2)}")

        result_diff = bRational(0.5,1.5)
        result_MP = compute_Pi(0.5,1.5)
        print(f"Integral from 0.5 to 1.5: {result_diff}")
        print(f"MPMath result: {result_MP}")
        print(f"Error: {abs(result_diff - result_MP)}")


        # Example usage
        z_value = 12.  # Replace with the desired value of z
        result = 1+PlanckIntC(z_value)
        print(f"Rational polynomial evaluated at z = {z_value}: {result}")
        result2 = compute_Pi(0, z_value)
        print(f"MPMath result: {result2}")
        print(f"Error: {abs(result - result2)}")


        result_diff = bRational(4,z_value)
        result_MP = compute_Pi(4,z_value)
        print(f"Integral from 4 to {z_value}: {result_diff}")
        print(f"MPMath result: {result_MP}")
        print(f"Error: {abs(result_diff - result_MP)}")

        result_diff = bRational(19.9,20)
        result_MP = 2.6501765584645523322e-7
        print(f"Integral from 19.9 to 20: {result_diff}")
        print(f"MPMath result: {result_MP}")
        print(f"Error: {abs(result_diff - result_MP)}")

        result_diff = bRational(19.9,50)
        result_MP = 3.2250573960478671454e-6
        print(f"Integral from 19.9 to 50: {result_diff}")
        print(f"MPMath result: {result_MP}")
        print(f"Error: {abs(result_diff - result_MP)}")

        #test parallel
        x = np.zeros(101)
        x[1:] = np.logspace(-1,math.log10(40),100)
        result_diff = bRationalParallel(x,100)
        result_MP = compute_PiParallel(x,100)
        print(f"Max error: {np.max(np.abs(result_diff - result_MP))}")

    except Exception as e:
        print(f"Error in evaluate_polynomial: {e}")