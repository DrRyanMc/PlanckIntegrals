import numpy as np
import math
from numba import njit

# Horner's method for evaluating P and Q
@njit(fastmath=True)
def horner(z, coeffs):
    result = 0.0
    for coeff in coeffs[::-1]:
        result = result * z + coeff
    return result
@njit(fastmath=True)
def li2l(x):
    """
    Compute the dilogarithm (Li_2) using the translated C implementation.
    This is a Python translation Alexander Voight's version of the dilogarithm
    https://github.com/Expander/polylogarithm/blob/master/src/c/Li2.c

    Parameters:
        x (float): The input value for the dilogarithm.
    
    Returns:
        float: The computed value of Li_2(x).
    """
    # Constants
    PI = 3.14159265358979323846264338327950288

    # Coefficients for P and Q (short precision version for demonstration)
    P = np.array([
        1.07061055633093042767673531395124630e+0,
       -5.25056559620492749887983310693176896e+0,
        1.03934845791141763662532570563508185e+1,
       -1.06275187429164237285280053453630651e+1,
        5.95754800847361224707276004888482457e+0,
       -1.78704147549824083632603474038547305e+0,
        2.56952343145676978700222949739349644e-1,
       -1.33237248124034497789318026957526440e-2,
        7.91217309833196694976662068263629735e-5
    ])
    Q = np.array([
        1.00000000000000000000000000000000000e+0,
       -5.20360694854541370154051736496901638e+0,
        1.10984640257222420881180161591516337e+1,
       -1.24997590867514516374467903875677930e+1,
        7.97919868560471967115958363930214958e+0,
       -2.87732383715218390800075864637472768e+0,
        5.49210416881086355164851972523370137e-1,
       -4.73366369162599860878254400521224717e-2,
        1.23136575793833628711851523557950417e-3
    ])

    y, r, s = 0.0, 0.0, 1.0

    # Transform x to [0, 1/2)
    if x < -1:
        l = math.log1p(-x) #(1 - x)
        y = 1 / (1 - x)
        r = -PI**2 / 6 + l * (0.5 * l - math.log(-x))
        s = 1
    elif x == -1:
        return -PI**2 / 12
    elif x < 0:
        l = math.log1p(-x)
        y = x / (x - 1)
        r = -0.5 * l**2
        s = -1
    elif x == 0:
        return x
    elif x < 0.5:
        y = x
        r = 0
        s = 1
    elif x < 1:
        y = 1 - x
        r = PI**2 / 6 - math.log(x) * np.log1p(-x)
        s = -1
    elif x == 1:
        return PI**2 / 6
    elif x < 2:
        l = math.log(x)
        y = 1 - 1 / x
        r = PI**2 / 6 - l * (np.log(y) + 0.5 * l)
        s = 1
    else:
        l = math.log(x)
        y = 1 / x
        r = PI**2 / 3 - 0.5 * l**2
        s = -1

    z = y - 0.25



    p = horner(z, P)
    q = horner(z, Q)

    return r + s * y * p / q

@njit(fastmath=True)
def li3_neg(x):
    """
    Compute Li_3(x) for x in [-1, 0] using coefficients for the negative range.

    Parameters:
        x (float): Input value in [-1, 0].

    Returns:
        float: The computed trilogarithm Li_3(x).
    """
    cp = np.array([1.0, -2.0281801754117129576, 1.4364029887561718540, -0.42240680435713030268,
                   0.047296746450884096877, -0.0013453536579918419568])
    cq = np.array([1.0, -2.1531801754117049035, 1.6685134736461140517, -0.56684857464584544310,
                   0.081999463370623961084, -0.0040756048502924149389, 0.000034316398489103212699])

    x2 = x * x
    x4 = x2 * x2

    p = horner(x, cp)#cp[0] + x * cp[1] + x2 * (cp[2] + x * cp[3]) + x4 * (cp[4] + x * cp[5])
    q = horner(x,cq) #cq[0] + x * cq[1] + x2 * (cq[2] + x * cq[3]) + x4 * (cq[4] + x * cq[5] + x2 * cq[6])

    return x * p / q

@njit(fastmath=True)
def li3_pos(x):
    """
    Compute Li_3(x) for x in [0, 0.5] using coefficients for the positive range.

    Parameters:
        x (float): Input value in [0, 0.5].

    Returns:
        float: The computed trilogarithm Li_3(x).
    """
    cp = np.array([1.0, -2.5224717303769789628, 2.3204919140887894133, -0.93980973288965037869,
                   0.15728950200990509052, -0.0075485193983677071129])
    cq = np.array([1.0, -2.6474717303769836244, 2.6143888433492184741, -1.1841788297857667038,
                   0.24184938524793651120, -0.018220900115898156346, 0.00024927971540017376759])

    x2 = x * x
    x4 = x2 * x2

    p = horner(x, cp)#cp[0] + x * cp[1] + x2 * (cp[2] + x * cp[3]) + x4 * (cp[4] + x * cp[5])
    q = horner(x,cq) #cq[0] + x * cq[1] + x2 * (cq[2] + x * cq[3]) + x4 * (cq[4] + x * cq[5] + x2 * cq[6])

    return x * p / q

@njit(fastmath=True)
def li3(x):
    """
    Compute the real trilogarithm Li_3(x).
    The implementation is based on the C code by Alexander Voigt:
    https://github.com/Expander/polylogarithm/blob/master/src/c/Li3.c
    Parameters:
        x (float): Input value.

    Returns:
        float: The computed trilogarithm Li_3(x).
    """
    zeta2 = 1.6449340668482264  # ζ(2)
    zeta3 = 1.2020569031595943  # ζ(3)

    if x < -1:
        l = math.log(-x)
        return li3_neg(1 / x) - l * (zeta2 + (1 / 6) * l * l)
    elif x == -1:
        return -0.75 * zeta3
    elif x < 0:
        return li3_neg(x)
    elif x == 0:
        return 0.0
    elif x < 0.5:
        return li3_pos(x)
    elif x == 0.5:
        return 0.53721319360804020
    elif x < 1:
        l = np.log(x)
        return -li3_neg(1 - 1 / x) - li3_pos(1 - x) + zeta3 + l * (zeta2 + l * (-0.5 * math.log1p(-x) + (1 / 6) * l))
    elif x == 1:
        return zeta3
    elif x < 2:
        l = np.log(x)
        return -li3_neg(1 - x) - li3_pos(1 - 1 / x) + zeta3 + l * (zeta2 + l * (-0.5 * math.log(x - 1) + (1 / 6) * l))
    else:
        l = np.log(x)
        return li3_pos(1 / x) + l * (2 * zeta2 - (1 / 6) * l * l)

# Coefficients for the various ranges of x
@njit(fastmath=True)
def li4_neg(x):
    cp = np.array([1.0, -1.8532099956062184217, 1.1937642574034898249, -0.31817912243893560382,
                   0.032268284189261624841, -0.00083773570305913850724])
    cq = np.array([1.0, -1.9157099956062165688, 1.3011504531166486419, -0.37975653506939627186,
                   0.045822723996558783670, -0.0018023912938765272341, 0.000010199621542882314929])

    x2 = x * x
    x4 = x2 * x2

    p = horner(x,cp)#cp[0] + x * cp[1] + x2 * (cp[2] + x * cp[3]) + x4 * (cp[4] + x * cp[5])
    q = horner(x,cq)#cq[0] + x * cq[1] + x2 * (cq[2] + x * cq[3]) + x4 * (cq[4] + x * cq[5] + x2 * cq[6])

    return x * p / q

@njit(fastmath=True)
def li4_half(x):
    cp = np.array([1.0000000000000000414, -2.0588072418045364525, 1.4713328756794826579, -0.42608608613069811474,
                   0.042975084278851543150, -0.00068314031819918920802])
    cq = np.array([1.0, -2.1213072418045207223, 1.5915688992789175941, -0.50327641401677265813,
                   0.061467217495127095177, -0.0019061294280193280330])

    x2 = x * x
    x4 = x2 * x2

    p = horner(x,cp)#cp[0] + x * cp[1] + x2 * (cp[2] + x * cp[3]) + x4 * (cp[4] + x * cp[5])
    q = horner(x,cq)#cq[0] + x * cq[1] + x2 * (cq[2] + x * cq[3]) + x4 * (cq[4] + x * cq[5])

    return x * p / q

@njit(fastmath=True)
def li4_mid(x):
    cp = np.array([3.2009826406098890447e-9, 0.99999994634837574160, -2.9144851228299341318,
                   3.1891031447462342009, -1.6009125158511117090, 0.35397747039432351193, -0.025230024124741454735])
    cq = np.array([1.0, -2.9769855248411488460, 3.3628208295110572579, -1.7782471949702788393,
                   0.43364007973198649921, -0.039535592340362510549, 0.00057373431535336755591])

    x2 = x * x
    x4 = x2 * x2

    p = horner(x,cp)#cp[0] + x * cp[1] + x2 * (cp[2] + x * cp[3]) + x4 * (cp[4] + x * cp[5] + x2 * cp[6])
    q = horner(x,cq)#cq[0] + x * cq[1] + x2 * (cq[2] + x * cq[3]) + x4 * (cq[4] + x * cq[5] + x2 * cq[6])

    return p / q

@njit(fastmath=True)
def li4_one(x):
    zeta2 = 1.6449340668482264  # ζ(2)
    zeta3 = 1.2020569031595943  # ζ(3)
    zeta4 = 1.0823232337111382  # ζ(4)

    l = math.log(x)
    l2 = l * l

    return zeta4 + l * (zeta3 + l * (0.5 * zeta2 + l * (11.0 / 36 - (1.0 / 6) * math.log(-l) +
                     l * (-1.0 / 48 + l * (-1.0 / 1440 + l2 * (1.0 / 604800 - l2 / 91445760))))))

@njit(fastmath=True)
def li4(x):
    """
    Compute the real tetralogarithm Li_4(x).
    The implementation is based on the C code by Alexander Voigt:
    https://github.com/Expander/polylogarithm/blob/master/src/c/Li4.c
    Parameters:
        x (float): Input value.

    Returns:
        float: The computed tetralogarithm Li_4(x).
    """
    zeta2 = 1.6449340668482264  # ζ(2)
    zeta4 = 1.0823232337111382  # ζ(4)

    app, rest, sgn = 0.0, 0.0, 1.0

    if x < -1:
        l = np.log(-x)
        l2 = l * l
        x = 1 / x
        rest = -7.0 / 4 * zeta4 + l2 * (-0.5 * zeta2 - l2 / 24)
        sgn = -1
    elif x == -1:
        return -7.0 / 8 * zeta4
    elif x == 0:
        return 0.0
    elif x < 1:
        rest = 0.0
        sgn = 1.0
    elif x == 1:
        return zeta4
    else:
        l = math.log(x)
        l2 = l * l
        x = 1 / x
        rest = 2 * zeta4 + l2 * (zeta2 - l2 / 24)
        sgn = -1

    if x < 0:
        app = li4_neg(x)
    elif x < 0.5:
        app = li4_half(x)
    elif x < 0.8:
        app = li4_mid(x)
    else:
        app = li4_one(x)

    return rest + sgn * app

@njit(fastmath=True)
def compute_Pi_Poly(x):
    """
    Compute the value of Pi(x) based on the rational polylogs Li_2, Li_3, and Li_4.

    Parameters:
        x (float): The input value for the formula.

    Returns:
        float: The computed value of Pi(x).
    """
    if (np.abs(x)<1e-12):
        return -1.
    ex = math.exp(-x)
    x2 = x*x
    x3 = x*x2
    # Compute the terms
    term1 = -6 * li4(ex) # 6 * Li_4(e^(-x))
    term2 = -6 *x* li3(ex)  # 6 * Li_3(e^(-x))
    term3 = -3 * x2 * li2l(ex)  # 3x^2 * Li_2(e^(-x))
    term4 = x3 * math.log(1 -ex)  # x^3 * log(1 - e^(-x))
    
    # Combine terms with the coefficient
    result = 0.153989733820265 * (term1 + term2 + term3 + term4) #no + 1 because we take difference
    return result
def b_Poly(a,b):
    return compute_Pi_Poly(b) - compute_Pi_Poly(a)

if __name__ == "__main__":
    # Example Usage
    from mpmath import polylog, log, pi
    import matplotlib.pyplot as plt
    comp_val = li2l(0.5)
    print("Li_2(0.5) =", comp_val)
    mathematica_val = 0.582240526465012505905018636651572763823
    print("The value from Mathematica is", mathematica_val)
    print("The relative error is", abs(comp_val - mathematica_val) / mathematica_val)

    x_values = np.linspace(0, 10, 100)  # Example range of x values
    results = np.array([li3(np.exp(-x)) for x in x_values])
    #plt.plot(x_values,(results))
    results2 = np.array([polylog(3,np.exp(-x)) for x in x_values])
    print("The relative error is", np.max(np.abs(results - results2) / results2))
    plt.plot(x_values,np.abs(results-results2))
    plt.ylabel("Difference $\\mathrm{Li}_n(e^{-x})$")
    plt.xlabel("x")

    x_values = np.linspace(0, 10, 100)  # Example range of x values
    results = np.array([li4(np.exp(-x)) for x in x_values])
    #plt.plot(x_values,(results))
    results2 = np.array([polylog(4,np.exp(-x)) for x in x_values])
    plt.semilogy(x_values,np.abs(results-results2))
    plt.show()

    from MPMathIntegral import compute_Pi
    z_values = [0.9e-3,1.5, 2, 2.5]  # Example inputs
    results = [b_Poly(0,z) for z in z_values]

    for z, result in zip(z_values, results):
        print(f"Poly({z}) = {result}")
        result2 = compute_Pi(0,z)
        print(f"MPMath result: {result2}")
        print(f"Error: {abs(result - result2)}")