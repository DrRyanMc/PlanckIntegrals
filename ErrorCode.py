import matplotlib.pyplot as plt
import math
import numpy as np
import ClarkApprox
import PolyLog
import RationalApprox
import Zimmerman
import Goldin
import Quadrature
import MPMathIntegral

x = np.logspace(-1, np.log10(40), 200)
y = np.array([MPMathIntegral.compute_Pi(0, x_val) for x_val in x])
print(y)
y2 = np.array([ClarkApprox.clark_formula(0, x_val, N=21, L=10) for x_val in x])
ClarkError = np.abs(y - y2)/y
y3 = np.array([PolyLog.b_Poly(0, x_val) for x_val in x])
PolyError = np.abs(y - y3)/y
print(y3)
y4 = np.array([RationalApprox.bRational(0, x_val) for x_val in x])
RationalError = np.abs(y - y4)/y
y5 = np.array([Zimmerman.zimInt(0, x_val) for x_val in x])
ZimmermanError = np.abs(y - y5)/y
y6 = np.array([Goldin.bGoldin(0, x_val) for x_val in x])
GoldinError = np.abs(y - y6)/y
n = 2
y7 = np.array([Quadrature.gauss_legendre_quadrature(0, x_val, n, *np.polynomial.legendre.leggauss(n)) for x_val in x])
QuadratureError2 = np.abs(y - y7)/y
n=8
y8 = np.array([Quadrature.gauss_legendre_quadrature(0, x_val, n, *np.polynomial.legendre.leggauss(n)) for x_val in x])
QuadratureError8 = np.abs(y - y8)/y
n=16
y9 = np.array([Quadrature.gauss_legendre_quadrature(0, x_val, n, *np.polynomial.legendre.leggauss(n)) for x_val in x])
QuadratureError16 = np.abs(y - y9)/y

plt.loglog(x, ClarkError, label="Clark")
plt.loglog(x, PolyError, label="PolyLog")
plt.loglog(x, RationalError, label="Rational")
plt.loglog(x, ZimmermanError, label="Zimmerman")
plt.loglog(x, GoldinError, label="Goldin")
plt.loglog(x, QuadratureError2, label="Quad n=2")
plt.loglog(x, QuadratureError8, label="Quad n=8")
plt.loglog(x, QuadratureError16, label="Quad n=16")
plt.xlabel("z")
plt.ylabel("Relative Error")
plt.legend()
plt.show()
