import matplotlib.pyplot as plt
import math
import csv
import numpy as np
import ClarkApprox
import PolyLog
import RationalApprox
import Zimmerman
import Goldin
import Quadrature
import MPMathIntegral
from matplotlib.legend_handler import HandlerLine2D

class HandlerLineEmoji(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        line = super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)[0]
        # Add an emoji as text
        emoji = legend.get_figure().text(
            xdescent + width / 2, ydescent + height / 2, "☭", color="red",
            fontsize= 14, #fontsize, 
            ha='center', va='center', transform=trans
        )
        return [line, emoji]
class HandlerLineEmojiStar(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        line = super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)[0]
        # Add an emoji as text
        emoji = legend.get_figure().text(
            xdescent + width / 2, ydescent + height / 2, "✩", color="blue",
            fontsize= 14, #fontsize, 
            ha='center', va='center', transform=trans
        )
        return [line, emoji]
n_group_sets = 99
ClarkError_Median = np.zeros(n_group_sets-1)
ClarkError_Max = np.zeros(n_group_sets-1)
ClarkError_Max93 = np.zeros(n_group_sets-1)
PolyError_Median = np.zeros(n_group_sets-1)
PolyError_Max = np.zeros(n_group_sets-1)
RationalError_Median = np.zeros(n_group_sets-1)
RationalError_Max = np.zeros(n_group_sets-1)
ZimmermanError_Median = np.zeros(n_group_sets-1)
ZimmermanError_Max = np.zeros(n_group_sets-1)
GoldinError_Median = np.zeros(n_group_sets-1)
GoldinError_Max = np.zeros(n_group_sets-1)
QuadratureError2_Median = np.zeros(n_group_sets-1)
QuadratureError2_Max = np.zeros(n_group_sets-1)
QuadratureError8_Median = np.zeros(n_group_sets-1)
QuadratureError8_Max = np.zeros(n_group_sets-1)
QuadratureError16_Median = np.zeros(n_group_sets-1)
QuadratureError16_Max = np.zeros(n_group_sets-1)
QuadratureError64_Median = np.zeros(n_group_sets-1)
QuadratureError64_Max = np.zeros(n_group_sets-1)

ClarkErrGroup = np.zeros(n_group_sets-1)
Clark93ErrGroup = np.zeros(n_group_sets-1)
RationalErrGroup = np.zeros(n_group_sets-1)
PolyErrorGroup = np.zeros(n_group_sets-1)
ZimmermanErrGroup = np.zeros(n_group_sets-1)
GoldinErrGroup = np.zeros(n_group_sets-1)
QuadErr2Group = np.zeros(n_group_sets-1)
QuadErr8Group = np.zeros(n_group_sets-1)
QuadErr16Group = np.zeros(n_group_sets-1)
QuadErr64Group = np.zeros(n_group_sets-1)

np.set_printoptions(precision=15)

for i,num_groups in enumerate(range(3,n_group_sets+2)):
    x = np.zeros(num_groups+1)
    x[1:-1] = np.logspace(-1, np.log10(20), num_groups-1)
    x[-1] = 100.
    print(f"Number of energy groups: {num_groups}")
    print(f"Energy group bounds: {x}")
    #y = MPMathIntegral.compute_PiParallel(x,num_groups)
    # Define the filename
    filename = "./BenchmarkResults/Bgs_Ng" + str(num_groups)+".csv"

    # Load the third column into a NumPy array
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        y = [float(row[2]) for row in reader if len(row) > 2]  # Extract third column (index 2)

    #Clark
    y2 = ClarkApprox.clark_formula_parallel(x, num_groups, N=21, L=10)
    y2[-1] = 1-np.sum(y2[:-1])
    ClarkError_Max[i] = np.max((np.abs(y - y2)/y)[:-1])
    ClarkError_Median[i] = np.median(np.abs(y - y2)/y)
    ClarkErrGroup[i] = np.argmax(np.abs(y - y2)/y)
    y2 = ClarkApprox.clark_formula_parallel(x, num_groups, N=9, L=3)
    y2[-1] = 1-np.sum(y2[:-1])
    ClarkError_Max93[i] = np.max((np.abs(y - y2)/y)[:-1])
    #PolyLog
    y2 = PolyLog.b_PolyParallel(x, num_groups)
    y2[-1] = 1-np.sum(y2[:-1])
    PolyError_Max[i] = np.max(np.abs(y - y2)/y)
    PolyError_Median[i] = np.median(np.abs(y - y2)/y)
    PolyErrorGroup[i] = np.argmax(np.abs(y - y2)/y)
    if (num_groups==2):
        print(f"Polylog: {y2}")
    #Rational
    y2 = RationalApprox.bRationalParallel(x, num_groups)
    y2[-1] = 1-np.sum(y2[:-1])
    print(np.abs(y - y2)/y)
    RationalError_Max[i] = np.max((np.abs(y - y2)/y)[:-1])
    RationalError_Median[i] = np.median(np.abs(y - y2)/y)
    RationalErrGroup[i] = np.argmax((np.abs(y - y2)/y)[:-1])
    if (num_groups==2):
        print("x: ", x)
        print(f"Rational: {y2}")
        print(f"Exact:    {y2}")
    #Zimmerman
    y2 = Zimmerman.zimIntParallel(x, num_groups)
    y2[-1] = 1-np.sum(y2[:-1])
    ZimmermanError_Max[i] = np.max((np.abs(y - y2)/y)[:-1])
    ZimmermanError_Median[i] = np.median(np.abs(y - y2)/y)
    ZimmermanErrGroup[i] = np.argmax(np.abs(y - y2)/y)
    #Goldin
    y2 = Goldin.bGoldinParallel(x, num_groups)
    y2[-1] = 1-np.sum(y2[:-1])
    GoldinError_Max[i] = np.max((np.abs(y - y2)/y)[:-1])
    GoldinError_Median[i] = np.median(np.abs(y - y2)/y)
    GoldinErrGroup[i] = np.argmax(np.abs(y - y2)/y)
    #Quadrature
    n = 4
    y2 = Quadrature.gl_parallel(x, num_groups, n, *np.polynomial.legendre.leggauss(n))
    y2[-1] = 1-np.sum(y2[:-1])
    QuadratureError2_Max[i] = np.max((np.abs(y - y2)/y)[:-1])
    QuadratureError2_Median[i] = np.median(np.abs(y - y2)/y)
    QuadErr2Group[i] = np.argmax(np.abs(y - y2)/y)
    n = 8
    y2 = Quadrature.gl_parallel(x, num_groups, n, *np.polynomial.legendre.leggauss(n))
    y2[-1] = 1-np.sum(y2[:-1])
    QuadratureError8_Max[i] = np.max((np.abs(y - y2)/y)[:-1])
    QuadratureError8_Median[i] = np.median(np.abs(y - y2)/y)
    QuadErr8Group[i] = np.argmax(np.abs(y - y2)/y)
    n = 16
    y2 = Quadrature.gl_parallel(x, num_groups, n, *np.polynomial.legendre.leggauss(n))
    y2[-1] = 1-np.sum(y2[:-1])
    QuadratureError16_Max[i] = np.max((np.abs(y - y2)/y)[:-1])
    QuadratureError16_Median[i] = np.median(np.abs(y - y2)/y)
    QuadErr16Group[i] = np.argmax(np.abs(y - y2)/y)
    n=64
    y2 = Quadrature.gl_parallel(x, num_groups, n, *np.polynomial.legendre.leggauss(n))
    y2[-1] = 1-np.sum(y2[:-1])
    QuadratureError64_Max[i] = np.max((np.abs(y - y2)/y)[:-1])
    QuadratureError64_Median[i] = np.median(np.abs(y - y2)/y)
    QuadErr64Group[i] = np.argmax(np.abs(y - y2)/y)


x = np.arange(3,n_group_sets+2)
fig, ax = plt.subplots()
ax.plot(x, ClarkErrGroup+1,"-", label="Clark (21,10)")
ax.plot(x, PolyErrorGroup+1,"--", label="PolyLog")
ax.plot(x, RationalErrGroup+1,"-.", label="Rational")
plt.show()
#assert 0
fig, ax = plt.subplots()
ax.semilogy(x, ClarkError_Max,"-", label="Clark (21,10)")
ax.semilogy(x, PolyError_Max,"--", label="PolyLog")
ax.semilogy(x, RationalError_Max,"-.", label="Rational")
plt.xlabel("Number of Energy Groups")
plt.ylabel("Relative Error")
ax.set_xlim(1,n_group_sets+2)
#ax.set_ylim(1e-14,1e-3)
custom_ticks = [2,10,20,30,40,50,60,70,80,90,100]
custom_labels = [r"$2$",r"$10$",r"$20$",r"$30$",r"$40$",r"$50$",r"$60$",r"$70$",r"$80$",r"$90$",r"$100$"]
ax.set_xticks(custom_ticks)
ax.set_xticklabels(custom_labels)
ax.grid(True, which="both", linestyle="--", alpha=0.5, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(fontsize=12)
plt.savefig("BenchmarkResults/ClarkPolyRationalError.pdf")
#print("Max Error Clark: ", ClarkError_Max)
plt.show()

fig, ax = plt.subplots()
ax.semilogy(x, ClarkError_Max93,"k-", label="Clark (9,3)")
#ax.semilogy(x, GoldinError_Max,"--", label="Goldin")
line_goldin, = ax.semilogy(x, GoldinError_Max,color='gray', alpha=0.8, label="Goldin")

# Add emoji markers manually
for xs, y in zip(x[0:-1:10], GoldinError_Max[0:-1:10]):
    plt.text(xs, y, '☭', color="red", fontsize=14, ha='center', va='center')

line_zimmerman, = ax.semilogy(x, ZimmermanError_Max,"b-", label="Zimmerman")
for xs, y in zip(x[0:-1:10], ZimmermanError_Max[0:-1:10]):
    plt.text(xs, y, '✩', color="blue", fontsize=14, ha='center', va='center')
#ax.semilogy(x, QuadratureError2_Max,"-.", markersize=3, alpha=0.8, label="GL n=2")
plt.xlabel("Number of Energy Groups")
plt.ylabel("Relative Error")
ax.set_xlim(1,n_group_sets+2)
ax.set_ylim(1e-6,1.5)
custom_ticks = [2,10,20,30,40,50,60,70,80,90,100]
custom_labels = [r"$2$",r"$10$",r"$20$",r"$30$",r"$40$",r"$50$",r"$60$",r"$70$",r"$80$",r"$90$",r"$100$"]
ax.set_xticks(custom_ticks)
ax.set_xticklabels(custom_labels)
ax.grid(True, which="both", linestyle="--", alpha=0.5, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.legend()
plt.legend(handler_map={line_goldin: HandlerLineEmoji(), line_zimmerman:HandlerLineEmojiStar()}, fontsize=12)
plt.savefig("BenchmarkResults/ClarkGoldinZimmermanError.pdf")
#print("Max Error Clark: ", ClarkError_Max)
plt.show()

fig, ax = plt.subplots()
ax.semilogy(x, QuadratureError2_Max,"-", markersize=3, alpha=0.8, label="GL n=4")
ax.semilogy(x, QuadratureError8_Max,"--", markersize=3, alpha=0.8, label="GL n=8")
ax.semilogy(x, QuadratureError16_Max,"-.",markersize=3, alpha=0.8, label="GL n=16")
ax.semilogy(x, QuadratureError64_Max,":",markersize=3, alpha=0.8, label="GL n=64")
plt.xlabel("Number of Energy Groups")
plt.ylabel("Relative Error")
ax.set_xlim(1,n_group_sets+2)
#ax.set_ylim(1e-6,1.5)
custom_ticks = [2,10,20,30,40,50,60,70,80,90,100]
custom_labels = [r"$2$",r"$10$",r"$20$",r"$30$",r"$40$",r"$50$",r"$60$",r"$70$",r"$80$",r"$90$",r"$100$"]
ax.set_xticks(custom_ticks)
ax.set_xticklabels(custom_labels)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which="both", linestyle="--", alpha=0.5, linewidth=0.5)
#plt.legend()
plt.legend(handler_map={line_goldin: HandlerLineEmoji()}, fontsize=12)
plt.savefig("BenchmarkResults/QuadratureError.pdf")
#print("Max Error Clark: ", ClarkError_Max)
plt.show()

