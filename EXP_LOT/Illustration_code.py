import numpy as np
import ot
import ot.plot
import LinSinkhorn
import time
import matplotlib.pyplot as plt
import scipy
from matplotlib import gridspec
from sklearn.neighbors import KernelDensity
from ot.datasets import make_1D_gauss as gauss


def Illustrative_example(n, m, sigma_x1=1, sigma_x2=1, sigma_y1=2, sigma_y2=2, seed=49):
    xa = np.arange(n)
    a = np.exp(-((xa - int(n / 3)) ** 2) / sigma_x1 ** 2) + 1.5 * np.exp(
        -((xa - int(5 * n / 6)) ** 2) / sigma_x2 ** 2
    )
    a = a / np.sum(a)

    xb = np.arange(m)
    b = 2 * np.exp(-((xb - int(m / 5)) ** 2) / sigma_y1 ** 2) + np.exp(
        -((xb - int(2 * m / 4)) ** 2) / sigma_y2 ** 2
    )
    b = b / np.sum(b)

    return a, b


Square_Euclidean_cost = lambda X, Y: LinSinkhorn.Square_Euclidean_Distance(X, Y)
Square_Euclidean_factorized_cost = lambda X, Y: LinSinkhorn.factorized_square_Euclidean(
    X, Y
)
Euclidean_cost = lambda X, Y: LinSinkhorn.Euclidean_Distance(X, Y)
L1_cost = lambda X, Y: LinSinkhorn.Lp_Distance(X, Y, p=1)
Lp_cost = lambda X, Y: LinSinkhorn.Lp_Distance(X, Y, p=1.5)

cost = Lp_cost

##########  Plot Coupling in 1D  ########
n, m = 200, 220

X, Y = np.arange(n).reshape(-1, 1), np.arange(m).reshape(-1, 1)
a, b = Illustrative_example(n, m, sigma_x1=25, sigma_x2=15, sigma_y1=30, sigma_y2=35)


### Setting with samples from distributions ###
C = cost(X, Y)
C = C / np.max(C)


list_P_Sin = []
regs = [0.001, 0.005, 0.05]

v_max = 0
v_min = 1
for reg in regs:
    res, acc_Sin, times_Sin, P_Sin = LinSinkhorn.Sinkhorn(
        C, reg, a, b, max_iter=1000, delta=1e-9, lam=0, time_out=200
    )
    v_min = min(np.min(P_Sin), v_min)
    v_max = max(np.max(P_Sin), v_max)
    list_P_Sin.append(P_Sin)


list_P_LOT = []
ranks = [3, 10, 50, 100]
gamma_init = "arbitrary"
gamma_0 = 100
method = "Dykstra"
alpha = 0.000001


reg_init = 1e-1
max_iter = 100
delta = 1e-9

C_init = True  # here the cost is a matrix
Init = "trivial"
seed_init = 49


for rank in ranks:
    res, acc_LOT, times_LOT, num_ops, Q, R, g = LinSinkhorn.Quad_LOT_MD(
        X,
        Y,
        a,
        b,
        rank,
        0,
        alpha,
        C,
        C_init=C_init,
        Init=Init,
        seed_init=seed_init,
        reg_init=reg_init,
        gamma_init=gamma_init,
        gamma_0=gamma_0,
        method=method,
        max_iter=max_iter,
        delta=delta,
        max_iter_IBP=10000,
        delta_IBP=1e-9,
        lam_IBP=0,
        time_out=100,
    )
    P_LOT = np.dot(Q / g, R.T)
    v_min = min(np.min(P_LOT), v_min)
    v_max = max(np.max(P_LOT), v_max)
    list_P_LOT.append(P_LOT)


#### Plot Figures ####

fig = plt.figure(constrained_layout=True, figsize=(15, 6))
widths = [1, 3, 3, 3, 3]  # ncols
heights = [1, 3, 3]  # nrows
gs = fig.add_gridspec(nrows=3, ncols=5, width_ratios=widths, height_ratios=heights)


f_ax1 = fig.add_subplot(gs[1, 0])
f_ax1.plot(a, X, "b", label="Source distribution")
f_ax1.axes.xaxis.set_visible(False)
plt.gca().invert_xaxis()
plt.axis("off")

f_ax2 = fig.add_subplot(gs[2, 0])
f_ax2.plot(a, X, "b", label="Source distribution")
f_ax2.axes.xaxis.set_visible(False)
plt.gca().invert_xaxis()
plt.axis("off")


for k in range(len(list_P_Sin)):

    f_ax3 = fig.add_subplot(gs[0, k + 1])
    f_ax3.plot(Y, b, "r", label="Target distribution")
    f_ax3.axes.xaxis.set_visible(False)
    plt.axis("off")

    num_Sin = len(list_P_Sin)
    if k == 0:
        f_ax4 = fig.add_subplot(gs[1, k + 1], sharex=f_ax3, sharey=f_ax1)
        f_ax4.imshow(
            list_P_Sin[-1 - k], interpolation="nearest", cmap="Greys", aspect="auto"
        )

        f_ax5 = fig.add_subplot(gs[2, k + 1], sharey=f_ax2)
        f_ax5.imshow(
            list_P_LOT[k], interpolation="nearest", cmap="Greys", aspect="auto"
        )
    else:
        f_ax4 = fig.add_subplot(gs[1, k + 1], sharex=f_ax3)
        f_ax4.imshow(
            list_P_Sin[-1 - k], interpolation="nearest", cmap="Greys", aspect="auto"
        )

        f_ax5 = fig.add_subplot(gs[2, k + 1])
        f_ax5.imshow(
            list_P_LOT[k], interpolation="nearest", cmap="Greys", aspect="auto"
        )

    f_ax4.set_xticks([])
    f_ax4.set_yticks([])

    f_ax5.set_xticks([])
    f_ax5.set_yticks([])

fig.tight_layout(pad=3.0)
fig.savefig("plot_coupling.pdf", bbox_inches="tight")
plt.show()
