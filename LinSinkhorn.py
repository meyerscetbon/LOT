import numpy as np
import time
from sklearn.cluster import KMeans
import scipy
from scipy import special


# Here C = C1 * C2 and P = P1 * P2
# Useful for kernel-based methods
def compute_OT(P1, P2, C1, C2):
    OT_trans_1 = np.dot(P1.T, C1)
    OT_trans_2 = np.dot(C2, P2.T)

    OT_trans = np.dot(OT_trans_1, OT_trans_2)
    res = np.trace(OT_trans)

    return res


def KL(A, B):
    Ratio_trans = np.log(A) - np.log(B)
    return np.sum(A * Ratio_trans)


def Sinkhorn(C, reg, a, b, max_iter=1000, delta=1e-3, lam=0, time_out=200):
    start = time.time()
    acc = []
    times = []

    # K = np.exp(-C/reg)
    # Next 3 lines equivalent to K= np.exp(-C/reg), but faster to compute
    K = np.empty(C.shape, dtype=C.dtype)
    np.divide(C, -reg, out=K)
    np.exp(K, out=K)

    v = np.ones(np.shape(b)[0])
    u_trans = np.dot(K, v) + lam  # add regularization to avoid divide 0

    err = 1
    n_iter = 0
    while n_iter < max_iter:
        if err > delta:
            n_iter = n_iter + 1

            # Update u
            u = a / u_trans

            # Update v
            v_trans = np.dot(K.T, u) + lam
            v = b / v_trans

            # Update the coupling
            P = u.reshape((-1, 1)) * K * v.reshape((1, -1))

            # Update the error
            u_trans = np.dot(K, v) + lam
            err = np.sum(np.abs(u * u_trans - a))

            # Update the total cost
            OT_trans = np.sum(P * C)
            if np.isnan(OT_trans) == True:
                print("Error: NaN OT value")
                return "Error"
            else:
                acc.append(OT_trans)
                end = time.time()
                tim_actual = end - start
                times.append(tim_actual)
                if tim_actual > time_out:
                    return acc[-1], np.array(acc), np.array(times), P
        else:
            return acc[-1], np.array(acc), np.array(times), P

    return acc[-1], np.array(acc), np.array(times), P


def Sinkhorn_LSE(C, reg, a, b, max_iter=1000, delta=1e-3, lam=0, time_out=200):
    start = time.time()
    acc = []
    times = []

    f = np.zeros(np.shape(a)[0])
    g = np.zeros(np.shape(b)[0])

    err = 1
    n_iter = 0
    while n_iter < max_iter:
        if err > delta:
            n_iter = n_iter + 1

            # Update f
            C_tilde = f[:, None] + g[None, :] - C
            C_tilde = C_tilde / reg
            f = reg * np.log(a) + f - reg * scipy.special.logsumexp(C_tilde, axis=1)

            # Update g
            C_tilde = f[:, None] + g[None, :] - C
            C_tilde = C_tilde / reg
            g = reg * np.log(b) + g - reg * scipy.special.logsumexp(C_tilde, axis=0)

            # Update the coupling
            C_tilde = f[:, None] + g[None, :] - C
            C_tilde = C_tilde / reg
            P = np.exp(C_tilde)

            # Update the error
            err = np.sum(np.abs(np.sum(P, axis=1) - a))

            # Update the total cost
            OT_trans = np.sum(P * C)
            if np.isnan(OT_trans) == True:
                print("Error: NaN OT value")
                return "Error"
            else:
                acc.append(OT_trans)
                end = time.time()
                tim_actual = end - start
                times.append(tim_actual)
                if tim_actual > time_out:
                    return acc[-1], np.array(acc), np.array(times), P
        else:
            return acc[-1], np.array(acc), np.array(times), P

    return acc[-1], np.array(acc), np.array(times), P


# Linear RF Sinkhorn: C = C1 * C2
def Lin_RF_Sinkhorn(C1, C2, reg, a, b, rank, seed=49, max_iter=1000, delta=1e-3, lam=0):
    start = time.time()
    acc = []
    times = []

    A, B = RF_Approx(-C1, C2, reg, num_samples=rank, seed=seed)

    v = np.ones(np.shape(b)[0])
    u_trans = np.dot(A, np.dot(B, v)) + lam

    err = 1
    n_iter = 0
    while n_iter < max_iter:
        if err > delta:
            n_iter = n_iter + 1

            # Update u
            u = a / u_trans

            # Update v
            v_trans = np.dot(B.T, np.dot(A.T, u)) + lam
            v = b / v_trans

            # Update the coupling
            P1 = u.reshape((-1, 1)) * A
            P2 = B * v.reshape((1, -1))

            # Update the error
            u_trans = np.dot(A, np.dot(B, v)) + lam
            err = np.sum(np.abs(u * u_trans - a))

            # Update total cost
            OT_trans = compute_OT(P1, P2, C1, C2)
            if np.isnan(OT_trans) == True:
                print("Error: NaN OT value")
                return "Error"
            else:
                acc.append(OT_trans)
                end = time.time()
                times.append(end - start)

        else:
            return acc[-1], np.array(acc), np.array(times), P1, P2

    return acc[-1], np.array(acc), np.array(times), P1, P2


# Linear Nys Sinkhorn: C = C1 * C2
def Lin_Nys_Sinkhorn(
    C1, C2, reg, a, b, rank, seed=49, max_iter=1000, delta=1e-3, lam=0
):
    start = time.time()
    acc = []
    times = []

    V1, V2 = Nys_approx(-C1, C2.T, reg, rank, seed=seed, stable=1e-10)
    A = np.dot(V2, np.linalg.inv(V1))
    A = A[: len(a), :]
    B = V2.T
    B = B[:, len(a) :]

    v = np.ones(np.shape(b)[0])
    u_trans = np.dot(A, np.dot(B, v)) + lam

    err = 1
    n_iter = 0
    while n_iter < max_iter:
        if err > delta:
            n_iter = n_iter + 1

            # Update u
            u = a / u_trans

            # Update v
            v_trans = np.dot(B.T, np.dot(A.T, u)) + lam
            v = b / v_trans

            # Update the coupling
            P1 = u.reshape((-1, 1)) * A
            P2 = B * v.reshape((1, -1))

            # Update the error
            u_trans = np.dot(A, np.dot(B, v)) + lam
            err = np.sum(np.abs(u * u_trans - a))

            # Update the total cost
            OT_trans = compute_OT(P1, P2, C1, C2)
            if np.isnan(OT_trans) == True:
                print("Error: NaN OT value")
                return "Error"
            else:
                acc.append(OT_trans)
                end = time.time()
                times.append(end - start)

        else:
            return acc[-1], np.array(acc), np.array(times), P1, P2

    return acc[-1], np.array(acc), np.array(times), P1, P2


def UpdateHubs(X, Y, gamma_1, gamma_2):
    Z = np.dot(gamma_1, X) + np.dot(gamma_2, Y)
    norm = np.sum(gamma_1 + gamma_2, axis=1)
    Z = (Z.T / norm).T

    return Z


# Here cost is a function
def UpdatePlans(X, Y, Z, a, b, reg, cost, max_iter=1000, delta=1e-9, lam=0):

    C1 = cost(Z, X)  # d * n * r
    K1 = np.exp(-C1 / reg)  # r x n

    C2 = cost(Z, Y)  # d * m * r
    K2 = np.exp(-C2 / reg)  # r x m

    r = np.shape(Z)[0]
    u1, u2 = np.ones(r), np.ones(r)
    v1, v2 = np.ones(np.shape(a)[0]), np.ones(np.shape(b)[0])

    v1_trans = np.dot(K1.T, u1)  # r * n
    v2_trans = np.dot(K2.T, u2)  # r * m

    w = np.ones(r) / r  # r

    err = 1
    n_iter = 0
    while n_iter < max_iter:
        u1_prev, v1_prev = u1, v1
        u2_prev, v2_prev = u2, v2
        w_prev = w
        if err > delta:
            n_iter = n_iter + 1

            # Update v1, v2
            v1 = a / v1_trans  # n
            u1_trans = np.dot(K1, v1)  # n * r

            v2 = b / v2_trans  # m
            u2_trans = np.dot(K2, v2)  # m * r

            # Update w
            w = (u1 * u1_trans * u2 * u2_trans) ** (1 / 2)  # 4 * r

            # Update u1, u2
            u1 = w / u1_trans  # r
            u2 = w / u2_trans  # r

            # Update the error
            v1_trans = np.dot(K1.T, u1)  # n * r
            err_1 = np.sum(np.abs(v1 * v1_trans - a))
            v2_trans = np.dot(K2.T, u2)  # n * r
            err_2 = np.sum(np.abs(v2 * v2_trans - b))
            err = err_1 + err_2

            if (
                np.any(np.isnan(u1))
                or np.any(np.isnan(v1))
                or np.any(np.isnan(u2))
                or np.any(np.isnan(v2))
                or np.any(np.isinf(u1))
                or np.any(np.isinf(v1))
                or np.any(np.isinf(u2))
                or np.any(np.isinf(v2))
            ):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print("Warning: numerical errors at iteration", n_iter)
                u1, v1 = u1_prev, v1_prev
                u2, v2 = u2_prev, v2_prev
                w = w_prev
                break
        else:
            gamma_1 = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
            gamma_2 = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))

            return gamma_1, gamma_2, w

    gamma_1 = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
    gamma_2 = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))

    return gamma_1, gamma_2, w


# Here cost is a function
def UpdatePlans_LSE(X, Y, Z, a, b, reg, cost, max_iter=1000, delta=1e-9, lam=0):

    C1 = cost(Z, X)
    C2 = cost(Z, Y)

    r = np.shape(Z)[0]
    f1, f2 = np.zeros(r), np.zeros(r)
    g1, g2 = np.zeros(np.shape(a)[0]), np.zeros(np.shape(b)[0])

    w = np.ones(r) / r

    err = 1
    n_iter = 0
    while n_iter < max_iter:
        f1_prev, g1_prev = f1, g1
        f2_prev, g2_prev = f2, g2
        w_prev = w
        if err > delta:
            n_iter = n_iter + 1

            # Update g1, g2
            C1_tilde = (
                f1.reshape(-1, 1) * np.ones((1, np.shape(a)[0]))
                + np.ones((r, 1)) * g1.reshape(1, -1)
                - C1
            )
            C1_tilde = C1_tilde / reg
            g1 = reg * np.log(a) + g1 - reg * scipy.special.logsumexp(C1_tilde, axis=0)

            C2_tilde = (
                f2.reshape(-1, 1) * np.ones((1, np.shape(b)[0]))
                + np.ones((r, 1)) * g2.reshape(1, -1)
                - C2
            )
            C2_tilde = C2_tilde / reg
            g2 = reg * np.log(b) + g2 - reg * scipy.special.logsumexp(C2_tilde, axis=0)

            # Update w
            C1_tilde = (
                f1.reshape(-1, 1) * np.ones((1, np.shape(a)[0]))
                + np.ones((r, 1)) * g1.reshape(1, -1)
                - C1
            )
            C1_tilde = C1_tilde / reg
            P1 = np.exp(C1_tilde)

            C2_tilde = (
                f2.reshape(-1, 1) * np.ones((1, np.shape(b)[0]))
                + np.ones((r, 1)) * g2.reshape(1, -1)
                - C2
            )
            C2_tilde = C2_tilde / reg
            P2 = np.exp(C2_tilde)

            w = (np.sum(P1, axis=1) * np.sum(P2, axis=1)) ** (1 / 2)
            log_w = (1 / 2) * (
                scipy.special.logsumexp(C1_tilde, axis=1)
                + scipy.special.logsumexp(C2_tilde, axis=1)
            )

            # Update f1, f2
            C1_tilde = (
                f1.reshape(-1, 1) * np.ones((1, np.shape(a)[0]))
                + np.ones((r, 1)) * g1.reshape(1, -1)
                - C1
            )
            C1_tilde = C1_tilde / reg
            f1 = reg * log_w + f1 - reg * scipy.special.logsumexp(C1_tilde, axis=1)

            C2_tilde = (
                f2.reshape(-1, 1) * np.ones((1, np.shape(b)[0]))
                + np.ones((r, 1)) * g2.reshape(1, -1)
                - C2
            )
            C2_tilde = C2_tilde / reg
            f2 = reg * log_w + f2 - reg * scipy.special.logsumexp(C2_tilde, axis=1)

            # Update the coupling P1, P2
            C1_tilde = (
                f1.reshape(-1, 1) * np.ones((1, np.shape(a)[0]))
                + np.ones((r, 1)) * g1.reshape(1, -1)
                - C1
            )
            C1_tilde = C1_tilde / reg
            P1 = np.exp(C1_tilde)

            C2_tilde = (
                f2.reshape(-1, 1) * np.ones((1, np.shape(b)[0]))
                + np.ones((r, 1)) * g2.reshape(1, -1)
                - C2
            )
            C2_tilde = C2_tilde / reg
            P2 = np.exp(C2_tilde)

            # Update the error
            err_1 = np.sum(np.abs(np.sum(P1, axis=0) - a))
            err_2 = np.sum(np.abs(np.sum(P2, axis=0) - b))
            err = err_1 + err_2

            if (
                np.any(np.isnan(f1))
                or np.any(np.isnan(g1))
                or np.any(np.isnan(f2))
                or np.any(np.isnan(g2))
                or np.any(np.isinf(f1))
                or np.any(np.isinf(g1))
                or np.any(np.isinf(f2))
                or np.any(np.isinf(g2))
            ):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print("Warning: numerical errors at iteration", n_iter)
                f1, g1 = f1_prev, g1_prev
                f2, g2 = f2_prev, g2_prev
                w = w_prev
                break
        else:
            return P1, P2, w

    # Update the coupling P1, P2
    C1_tilde = (
        f1.reshape(-1, 1) * np.ones((1, np.shape(a)[0]))
        + np.ones((r, 1)) * g1.reshape(1, -1)
        - C1
    )
    C1_tilde = C1_tilde / reg
    P1 = np.exp(C1_tilde)

    C2_tilde = (
        f2.reshape(-1, 1) * np.ones((1, np.shape(b)[0]))
        + np.ones((r, 1)) * g2.reshape(1, -1)
        - C2
    )
    C2_tilde = C2_tilde / reg
    P2 = np.exp(C2_tilde)

    return P1, P2, w


# Same as UpdatePlans where the inputs are no more vectors but rather matrices
def UpdatePlans_Matrix(C1, C2, a, b, reg, max_iter=1000, delta=1e-9, lam=0):
    K1 = np.exp(-C1.T / reg)  # size: r x n
    K2 = np.exp(-C2 / reg)  # size: r x m

    r = np.shape(C1)[1]
    u1, u2 = np.ones(r), np.ones(r)
    v1, v2 = np.ones(np.shape(a)[0]), np.ones(np.shape(b)[0])

    v1_trans = np.dot(K1.T, u1)
    v2_trans = np.dot(K2.T, u2)

    w = np.ones(r) / r

    err = 1
    n_iter = 0
    while n_iter < max_iter:
        u1_prev, v1_prev = u1, v1
        u2_prev, v2_prev = u2, v2
        w_prev = w
        if err > delta:
            n_iter = n_iter + 1

            # Update v1, v2
            v1 = a / v1_trans
            u1_trans = np.dot(K1, v1)

            v2 = b / v2_trans
            u2_trans = np.dot(K2, v2)

            # Update w
            w = (u1 * u1_trans * u2 * u2_trans) ** (1 / 2)

            # Update u1, u2
            u1 = w / u1_trans
            u2 = w / u2_trans

            # Update the error
            v1_trans = np.dot(K1.T, u1)
            err_1 = np.sum(np.abs(v1 * v1_trans - a))
            v2_trans = np.dot(K2.T, u2)
            err_2 = np.sum(np.abs(v2 * v2_trans - b))
            err = err_1 + err_2

            if (
                np.any(np.isnan(u1))
                or np.any(np.isnan(v1))
                or np.any(np.isnan(u2))
                or np.any(np.isnan(v2))
                or np.any(np.isinf(u1))
                or np.any(np.isinf(v1))
                or np.any(np.isinf(u2))
                or np.any(np.isinf(v2))
            ):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print("Warning: numerical errors at iteration", n_iter)
                u1, v1 = u1_prev, v1_prev
                u2, v2 = u2_prev, v2_prev
                w = w_prev
                break
        else:
            gamma_1 = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
            gamma_2 = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))
            return gamma_1.T, gamma_2.T, w

    gamma_1 = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
    gamma_2 = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))
    return gamma_1.T, gamma_2.T, w


# Here cost is a function: only the Squared Euclidean is legal
def FactoredOT(
    X,
    Y,
    a,
    b,
    reg,
    rank,
    cost,
    max_iter=1000,
    delta=1e-3,
    max_iter_Update=1000,
    delta_Update=1e-9,
    lam_Update=0,
    LSE=True,
    time_out=200,
):
    start = time.time()
    acc = []
    times = []

    C = cost(X, Y)

    kmeans = KMeans(n_clusters=rank, random_state=0).fit(X)
    Z = kmeans.cluster_centers_

    w = np.ones(rank) / rank
    gamma1 = w.reshape((-1, 1)) * a.reshape((1, -1))
    gamma2 = w.reshape((-1, 1)) * b.reshape((1, -1))

    err = 1
    niter = 0
    while niter < max_iter:
        gamma1_prev = gamma1
        gamma2_prev = gamma2
        w_prev = w
        if err > delta:
            niter = niter + 1
            if LSE == False:
                gamma1, gamma2, w = UpdatePlans(
                    X,
                    Y,
                    Z,
                    a,
                    b,
                    reg,
                    cost,
                    max_iter=max_iter_Update,
                    delta=delta_Update,
                    lam=lam_Update,
                )
            else:
                gamma1, gamma2, w = UpdatePlans_LSE(
                    X,
                    Y,
                    Z,
                    a,
                    b,
                    reg,
                    cost,
                    max_iter=max_iter_Update,
                    delta=delta_Update,
                    lam=lam_Update,
                )

            # Update the Hubs
            Z = UpdateHubs(X, Y, gamma1, gamma2)

            #  Update the total cost
            C_trans = np.dot(C, gamma2.T)
            C_trans = C_trans / w
            G = np.dot(gamma1, C_trans)
            OT_trans = np.trace(G)

            if niter > 10:
                ## Update the error
                err = np.abs(OT_trans - acc[-1]) / acc[-1]

                if np.isnan(err):
                    print("Error computation of the stopping criterion", niter)
                    gamma1 = gamma1_prev
                    gamma2 = gamma2_prev
                    w = w_prev
                    break

            if np.isnan(OT_trans) == True:
                print("Error: NaN OT value")
                return "Error"

            else:
                acc.append(OT_trans)
                end = time.time()
                tim_actual = end - start
                times.append(tim_actual)
                if tim_actual > time_out:
                    return (
                        acc[-1],
                        np.array(acc),
                        np.array(times),
                        gamma1.T,
                        gamma2.T,
                        w,
                    )
        else:
            return acc[-1], np.array(acc), np.array(times), gamma1.T, gamma2.T, w

    return acc[-1], np.array(acc), np.array(times), gamma1.T, gamma2.T, w


def LR_Dykstra_Sin(K1, K2, K3, a, b, alpha, max_iter=1000, delta=1e-9, lam=0):
    Q = K1
    R = K2
    g_old = K3

    r = np.shape(K3)[0]
    v1_old, v2_old = np.ones(r), np.ones(r)
    u1, u2 = np.ones(np.shape(a)[0]), np.ones(np.shape(b)[0])

    q_gi, q_gp = np.ones(r), np.ones(r)
    q_Q, q_R = np.ones(r), np.ones(r)

    err = 1
    n_iter = 0
    while n_iter < max_iter:
        u1_prev, v1_prev = u1, v1_old
        u2_prev, v2_prev = u2, v2_old
        g_prev = g_old
        if err > delta:
            n_iter = n_iter + 1

            # First Projection
            u1 = a / (np.dot(K1, v1_old) + lam)
            u2 = b / (np.dot(K2, v2_old) + lam)
            g = np.maximum(alpha, g_old * q_gi)
            q_gi = (g_old * q_gi) / (g + lam)
            g_old = g.copy()

            # Second Projection
            v1_trans = np.dot(K1.T, u1)
            v2_trans = np.dot(K2.T, u2)
            g = (g_old * q_gp * v1_old * q_Q * v1_trans * v2_old * q_R * v2_trans) ** (
                1 / 3
            )
            v1 = g / (v1_trans + lam)
            v2 = g / (v2_trans + lam)
            q_gp = (g_old * q_gp) / (g + lam)
            q_Q = (q_Q * v1_old) / (v1 + lam)
            q_R = (q_R * v2_old) / (v2 + lam)
            v1_old = v1.copy()
            v2_old = v2.copy()
            g_old = g.copy()

            # Update the error
            u1_trans = np.dot(K1, v1)
            err_1 = np.sum(np.abs(u1 * u1_trans - a))
            u2_trans = np.dot(K2, v2)
            err_2 = np.sum(np.abs(u2 * u2_trans - b))
            err = err_1 + err_2

            if (
                np.any(np.isnan(u1))
                or np.any(np.isnan(v1))
                or np.any(np.isnan(u2))
                or np.any(np.isnan(v2))
                or np.any(np.isinf(u1))
                or np.any(np.isinf(v1))
                or np.any(np.isinf(u2))
                or np.any(np.isinf(v2))
            ):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print("Warning: numerical errors at iteration", n_iter)
                u1, v1 = u1_prev, v1_prev
                u2, v2 = u2_prev, v2_prev
                g = g_prev
                break
        else:
            Q = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
            R = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))
            return Q, R, g

    Q = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
    R = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))

    return Q, R, g


def LR_Dykstra_LSE_Sin(
    C1, C2, C3, a, b, alpha, gamma, max_iter=1000, delta=1e-9, lam=0
):

    h_old = C3
    r = np.shape(C3)[0]
    g1_old, g2_old = np.zeros(r), np.zeros(r)
    f1, f2 = np.zeros(np.shape(a)[0]), np.zeros(np.shape(b)[0])

    w_gi, w_gp = np.zeros(r), np.zeros(r)
    w_Q, w_R = np.zeros(r), np.zeros(r)

    err = 1
    n_iter = 0
    while n_iter < max_iter:
        f1_prev, g1_prev = f1, g1_old
        f2_prev, g2_prev = f2, g2_old
        h_prev = h_old
        if err > delta:
            n_iter = n_iter + 1

            # First Projection
            C1_tilde = f1[:, None] + g1_old[None, :] - C1  # 2 * n * r
            C1_tilde = C1_tilde * gamma  # n * r
            f1 = (
                (1 / gamma) * np.log(a)
                + f1
                - (1 / gamma) * scipy.special.logsumexp(C1_tilde, axis=1)
            )  # 2 * n + 2 * n + n * r

            C2_tilde = f2[:, None] + g2_old[None, :] - C2  # 2 * m * r
            C2_tilde = C2_tilde * gamma  # m * r
            f2 = (
                (1 / gamma) * np.log(b)
                + f2
                - (1 / gamma) * scipy.special.logsumexp(C2_tilde, axis=1)
            )  # 2 * m + 2 * m + m * r

            h = h_old + w_gi  # 2 * r
            h = np.maximum((np.log(alpha) / gamma), h)  # r
            w_gi = h_old + w_gi - h  # 2 * r
            h_old = h.copy()

            # Update couplings
            C1_tilde = f1[:, None] + g1_old[None, :] - C1  # 2 * n * r
            C1_tilde = C1_tilde * gamma  # n * r
            alpha_1_trans = scipy.special.logsumexp(C1_tilde, axis=0)  # n * r

            C2_tilde = f2[:, None] + g2_old[None, :] - C2  # 2 * m * r
            C2_tilde = C2_tilde * gamma  # m * r
            alpha_2_trans = scipy.special.logsumexp(C2_tilde, axis=0)  # m * r

            # Second Projection
            h = (1 / 3) * (h_old + w_gp + w_Q + w_R)  # 4 * r
            h = h + (1 / (3 * gamma)) * alpha_1_trans  # 2 * r
            h = h + (1 / (3 * gamma)) * alpha_2_trans  # 2 * r
            g1 = h + g1_old - (1 / gamma) * alpha_1_trans  # 3 * r
            g2 = h + g2_old - (1 / gamma) * alpha_2_trans  # 3 * r

            w_Q = w_Q + g1_old - g1  # 2 * r
            w_R = w_R + g2_old - g2  # 2 * r
            w_gp = h_old + w_gp - h  # 2 * r

            g1_old = g1.copy()
            g2_old = g2.copy()
            h_old = h.copy()

            # Update couplings
            C1_tilde = f1[:, None] + g1_old[None, :] - C1  # 2 * n * r
            C1_tilde = C1_tilde * gamma  # n * r
            Q = np.exp(C1_tilde)  # n * r

            C2_tilde = f2[:, None] + g2_old[None, :] - C2  # 2 * n * r
            C2_tilde = C2_tilde * gamma  # n * r
            R = np.exp(C2_tilde)  # n * r

            g = np.exp(gamma * h)  # 2 * r

            # Update the error
            err_1 = np.sum(np.abs(np.sum(Q, axis=1) - a))
            err_2 = np.sum(np.abs(np.sum(R, axis=1) - b))
            err = err_1 + err_2

            if (
                np.any(np.isnan(f1))
                or np.any(np.isnan(g1))
                or np.any(np.isnan(f2))
                or np.any(np.isnan(g2))
                or np.any(np.isinf(f1))
                or np.any(np.isinf(g1))
                or np.any(np.isinf(f2))
                or np.any(np.isinf(g2))
            ):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print("Warning: numerical errors at iteration", n_iter)
                f1, g1 = f1_prev, g1_prev
                f2, g2 = f2_prev, g2_prev
                h = h_prev

                # Update couplings
                C1_tilde = f1[:, None] + g1_old[None, :] - C1
                C1_tilde = C1_tilde * gamma
                Q = np.exp(C1_tilde)

                C2_tilde = f2[:, None] + g2_old[None, :] - C2
                C2_tilde = C2_tilde * gamma
                R = np.exp(C2_tilde)

                g = np.exp(gamma * h)
                return Q, R, g

        else:
            return Q, R, g

    return Q, R, g


def LR_IBP_Sin(K1, K2, K3, a, b, max_iter=1000, delta=1e-9, lam=0):
    Q = K1
    R = K2
    g = K3

    r = np.shape(K3)[0]
    v1, v2 = np.ones(r), np.ones(r)
    u1, u2 = np.ones(np.shape(a)[0]), np.ones(np.shape(a)[0])

    u1_trans = np.dot(K1, v1)  # n * r
    u2_trans = np.dot(K2, v2)  # m * r

    err = 1
    n_iter = 0
    while n_iter < max_iter:
        u1_prev, v1_prev = u1, v1
        u2_prev, v2_prev = u2, v2
        g_prev = g
        if err > delta:
            n_iter = n_iter + 1

            # Update u1
            u1 = a / u1_trans  # n
            v1_trans = np.dot(K1.T, u1)  # n * r

            # Update u2
            u2 = a / u2_trans  # m
            v2_trans = np.dot(K2.T, u2)  # m * r

            # Update g
            g = (g * v1 * v1_trans * v2 * v2_trans) ** (1 / 3)  # 5 * r

            # Update v1
            v1 = g / v1_trans  # r

            # Update v2
            v2 = g / v2_trans  # r

            # Update the error
            u1_trans = np.dot(K1, v1)
            err_1 = np.sum(np.abs(u1 * u1_trans - a))
            u2_trans = np.dot(K2, v2)
            err_2 = np.sum(np.abs(u2 * u2_trans - b))
            err = err_1 + err_2

            if (
                np.any(np.isnan(u1))
                or np.any(np.isnan(v1))
                or np.any(np.isnan(u2))
                or np.any(np.isnan(v2))
                or np.any(np.isinf(u1))
                or np.any(np.isinf(v1))
                or np.any(np.isinf(u2))
                or np.any(np.isinf(v2))
            ):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print("Warning: numerical errors at iteration", n_iter)
                u1, v1 = u1_prev, v1_prev
                u2, v2 = u2_prev, v2_prev
                g = g_prev
                break
        else:
            Q = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
            R = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))
            return Q, R, g

    Q = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
    R = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))
    return Q, R, g


# If C_init == True: cost is the cost matrix
# If C_init == False: cost is the cost function
# Init == 'trivial', 'random', 'kmeans
def Quad_LOT_MD(
    X,
    Y,
    a,
    b,
    rank,
    reg,
    alpha,
    cost,
    Init="trivial",
    seed_init=49,
    C_init=False,
    reg_init=1e-3,
    gamma_init="theory",
    gamma_0=1e-1,
    method="IBP",
    max_iter=1000,
    delta=1e-3,
    max_iter_IBP=1000,
    delta_IBP=1e-9,
    lam_IBP=0,
    time_out=200,
):
    start = time.time()
    acc = []
    times = []

    n, m = np.shape(a)[0], np.shape(b)[0]

    if C_init == False:
        C = cost(X, Y)
    else:
        C = cost

    if len(C) == 2:
        print("Error: cost not adapted")
        return "Error"

    #### Initialization #####
    if Init == "kmeans":
        ## Init with K-means
        g = np.ones(rank) / rank
        kmeans = KMeans(n_clusters=rank, random_state=0).fit(X)
        Z = kmeans.cluster_centers_
        gamma1, gamma2, g = UpdatePlans(
            X,
            Y,
            Z,
            a,
            b,
            reg_init,
            cost,
            max_iter=max_iter_IBP,
            delta=delta_IBP,
            lam=lam_IBP,
        )
        Q, R = gamma1.T, gamma2.T

    # Init random
    if Init == "random":
        np.random.seed(seed_init)
        g = np.abs(np.random.randn(rank))
        g = g + 1  # r
        g = g / np.sum(g)  # r

        seed_init = seed_init + 1000
        np.random.seed(seed_init)
        Q = np.abs(np.random.randn(n, rank))
        Q = Q + 1  # n * r
        Q = (Q.T * (a / np.sum(Q, axis=1))).T  # n + n * r

        seed_init = seed_init + 1000
        np.random.seed(seed_init)
        R = np.abs(np.random.randn(m, rank))
        R = R + 1  # m * r
        R = (R.T * (b / np.sum(R, axis=1))).T  # m + m * r

    ###  Trivial Init
    if Init == "trivial":
        g = np.ones(rank) / rank  # r
        lambda_1 = min(np.min(a), np.min(g), np.min(b)) / 2

        a1 = np.arange(1, np.shape(a)[0] + 1)
        a1 = a1 / np.sum(a1)  # n
        a2 = (a - lambda_1 * a1) / (1 - lambda_1)  # 2 * n

        b1 = np.arange(1, np.shape(b)[0] + 1)
        b1 = b1 / np.sum(b1)  # m
        b2 = (b - lambda_1 * b1) / (1 - lambda_1)  # 2 * m

        g1 = np.arange(1, rank + 1)
        g1 = g1 / np.sum(g1)  # r
        g2 = (g - lambda_1 * g1) / (1 - lambda_1)  # 2 * r

        Q = lambda_1 * np.dot(a1[:, None], g1.reshape(1, -1)) + (1 - lambda_1) * np.dot(
            a2[:, None], g2.reshape(1, -1)  # 4 * n * r
        )
        R = lambda_1 * np.dot(b1[:, None], g1.reshape(1, -1)) + (1 - lambda_1) * np.dot(
            b2[:, None], g2.reshape(1, -1)  # 4 * m * r
        )

    if gamma_init == "theory":
        L_trans = (2 / (alpha) ** 4) * (np.linalg.norm(C) ** 2)
        L_trans = L_trans + ((reg + 2 * np.linalg.norm(C)) / (alpha ** 3)) ** 2
        L = np.sqrt(3 * L_trans)
        gamma = 1 / L

    if gamma_init == "regularization":
        gamma = 1 / reg

    if gamma_init == "arbitrary":
        gamma = gamma_0

    err = 1
    niter = 0
    while niter < max_iter:
        Q_prev = Q
        R_prev = R
        g_prev = g
        if err > delta:
            niter = niter + 1

            K1_trans_0 = np.dot(C, R)  # n * m * r
            C1_trans = K1_trans_0 / g + (reg - (1 / gamma)) * np.log(Q)  # 3 * n * r

            K2_trans_0 = np.dot(C.T, Q)  # m * n * r
            C2_trans = K2_trans_0 / g + (reg - (1 / gamma)) * np.log(R)  # 3 * m * r

            omega = np.diag(np.dot(Q.T, K1_trans_0))  # r * n * r
            C3_trans = omega / (g ** 2) - (reg - (1 / gamma)) * np.log(g)  # 4 * r

            # Update the coupling
            if method == "IBP":
                K1 = np.exp((-gamma) * C1_trans)
                K2 = np.exp((-gamma) * C2_trans)
                K3 = np.exp(gamma * C3_trans)
                Q, R, g = LR_IBP_Sin(
                    K1,
                    K2,
                    K3,
                    a,
                    b,
                    max_iter=max_iter_IBP,
                    delta=delta_IBP,
                    lam=lam_IBP,
                )

            if method == "Dykstra":
                K1 = np.exp((-gamma) * C1_trans)
                K2 = np.exp((-gamma) * C2_trans)
                K3 = np.exp(gamma * C3_trans)
                Q, R, g = LR_Dykstra_Sin(
                    K1,
                    K2,
                    K3,
                    a,
                    b,
                    alpha,
                    max_iter=max_iter_IBP,
                    delta=delta_IBP,
                    lam=lam_IBP,
                )

            if method == "Dykstra_LSE":
                Q, R, g = LR_Dykstra_LSE_Sin(
                    C1_trans,
                    C2_trans,
                    C3_trans,
                    a,
                    b,
                    alpha,
                    gamma,
                    max_iter=max_iter_IBP,
                    delta=delta_IBP,
                    lam=lam_IBP,
                )

            # Update the total cost
            C_trans = np.dot(C, R)
            C_trans = C_trans / g
            G = np.dot(Q.T, C_trans)
            OT_trans = np.trace(G)

            if niter > 10:
                ## Update the error: Practical error
                err = np.abs(OT_trans - acc[-1]) / acc[-1]

                if np.isnan(err):
                    print("Error computation of the stopping criterion", niter)
                    Q = Q_prev
                    R = R_prev
                    g = g_prev
                    break

            if np.isnan(OT_trans) == True:
                print("Error: NaN OT value")
                return "Error"

            else:
                acc.append(OT_trans)
                end = time.time()
                tim_actual = end - start
                times.append(tim_actual)
                if tim_actual > time_out:
                    return (
                        acc[-1],
                        np.array(acc),
                        np.array(times),
                        Q,
                        R,
                        g,
                    )
        else:
            return (
                acc[-1],
                np.array(acc),
                np.array(times),
                Q,
                R,
                g,
            )

    return acc[-1], np.array(acc), np.array(times), Q, R, g


# gamma_init = 'theory', 'regularization', 'arbitrary'
# method = 'IBP', 'Dykstra', 'Dykstra_LSE'
# If C_init = True: cost_factorized = C1,C2
# If C_init = False: cost_factorized is a function
# Init = 'trivial', kmeans', 'random'
def Lin_LOT_MD(
    X,
    Y,
    a,
    b,
    rank,
    reg,
    alpha,
    cost,
    cost_factorized,
    Init="trivial",
    seed_init=49,
    C_init=False,
    reg_init=1e-1,
    gamma_init="theory",
    gamma_0=1e-1,
    method="IBP",
    max_iter=1000,
    delta=1e-3,
    max_iter_IBP=1000,
    delta_IBP=1e-9,
    lam_IBP=0,
    time_out=200,
):
    start = time.time()
    acc = []
    times = []

    n, m = np.shape(a)[0], np.shape(b)[0]

    if C_init == False:
        C = cost_factorized(X, Y)
        if len(C) == 2:
            C1, C2 = C
        else:
            print("Error: cost not adapted")
            return "Error"
    else:
        C1, C2 = cost_factorized

    n, d = np.shape(C1)

    ########### Initialization ###########

    ## Init with K-means
    if Init == "kmeans":
        g = np.ones(rank) / rank
        kmeans = KMeans(n_clusters=rank, random_state=0).fit(X)
        Z = kmeans.cluster_centers_
        gamma1, gamma2, g, count_op_Barycenter = UpdatePlans(
            X,
            Y,
            Z,
            a,
            b,
            reg_init,
            cost,
            max_iter=max_iter_IBP,
            delta=delta_IBP,
            lam=lam_IBP,
        )
        Q, R = gamma1.T, gamma2.T

    ## Init random
    if Init == "random":
        np.random.seed(seed_init)
        g = np.abs(np.random.randn(rank))
        g = g + 1
        g = g / np.sum(g)
        n, d = np.shape(X)
        m, d = np.shape(Y)

        seed_init = seed_init + 1000
        np.random.seed(seed_init)
        Q = np.abs(np.random.randn(n, rank))
        Q = Q + 1
        Q = (Q.T * (a / np.sum(Q, axis=1))).T

        seed_init = seed_init + 1000
        np.random.seed(seed_init)
        R = np.abs(np.random.randn(m, rank))
        R = R + 1
        R = (R.T * (b / np.sum(R, axis=1))).T

    ## Init trivial
    if Init == "trivial":
        g = np.ones(rank) / rank
        lambda_1 = min(np.min(a), np.min(g), np.min(b)) / 2

        a1 = np.arange(1, np.shape(a)[0] + 1)
        a1 = a1 / np.sum(a1)
        a2 = (a - lambda_1 * a1) / (1 - lambda_1)

        b1 = np.arange(1, np.shape(b)[0] + 1)
        b1 = b1 / np.sum(b1)
        b2 = (b - lambda_1 * b1) / (1 - lambda_1)

        g1 = np.arange(1, rank + 1)
        g1 = g1 / np.sum(g1)
        g2 = (g - lambda_1 * g1) / (1 - lambda_1)

        Q = lambda_1 * np.dot(a1[:, None], g1.reshape(1, -1)) + (1 - lambda_1) * np.dot(
            a2[:, None], g2.reshape(1, -1)
        )
        R = lambda_1 * np.dot(b1[:, None], g1.reshape(1, -1)) + (1 - lambda_1) * np.dot(
            b2[:, None], g2.reshape(1, -1)
        )
    #####################################

    if gamma_init == "theory":
        L_trans = (
            (2 / (alpha) ** 4) * (np.linalg.norm(C1) ** 2) * (np.linalg.norm(C1) ** 2)
        )
        L_trans = (
            L_trans
            + ((reg + 2 * np.linalg.norm(C1) * np.linalg.norm(C1)) / (alpha ** 3)) ** 2
        )
        L = np.sqrt(3 * L_trans)
        gamma = 1 / L
        print(gamma)

    if gamma_init == "regularization":
        gamma = 1 / reg

    if gamma_init == "arbitrary":
        gamma = gamma_0

    err = 1
    niter = 0
    while niter < max_iter:
        Q_prev = Q
        R_prev = R
        g_prev = g
        if err > delta:
            niter = niter + 1

            K1_trans_0 = np.dot(C2, R)  # d * m * r
            K1_trans_0 = np.dot(C1, K1_trans_0)  # n * d * r
            C1_trans = K1_trans_0 / g + (reg - (1 / gamma)) * np.log(Q)  # 3 * n * r

            K2_trans_0 = np.dot(C1.T, Q)  # d * n * r
            K2_trans_0 = np.dot(C2.T, K2_trans_0)  # m * d * r
            C2_trans = K2_trans_0 / g + (reg - (1 / gamma)) * np.log(R)  # 3 * m * r

            omega = np.diag(np.dot(Q.T, K1_trans_0))  # r * n * r
            C3_trans = (omega / (g ** 2)) - (reg - (1 / gamma)) * np.log(g)  # 4 * r

            # Update the coupling
            if method == "IBP":
                K1 = np.exp((-gamma) * C1_trans)
                K2 = np.exp((-gamma) * C2_trans)
                K3 = np.exp(gamma * C3_trans)
                Q, R, g = LR_IBP_Sin(
                    K1,
                    K2,
                    K3,
                    a,
                    b,
                    max_iter=max_iter_IBP,
                    delta=delta_IBP,
                    lam=lam_IBP,
                )

            if method == "Dykstra":
                K1 = np.exp((-gamma) * C1_trans)
                K2 = np.exp((-gamma) * C2_trans)
                K3 = np.exp(gamma * C3_trans)
                Q, R, g = LR_Dykstra_Sin(
                    K1,
                    K2,
                    K3,
                    a,
                    b,
                    alpha,
                    max_iter=max_iter_IBP,
                    delta=delta_IBP,
                    lam=lam_IBP,
                )

            if method == "Dykstra_LSE":
                Q, R, g = LR_Dykstra_LSE_Sin(
                    C1_trans,
                    C2_trans,
                    C3_trans,
                    a,
                    b,
                    alpha,
                    gamma,
                    max_iter=max_iter_IBP,
                    delta=delta_IBP,
                    lam=lam_IBP,
                )

            # Update the total cost

            # Metric used in the MIT paper: useless
            # OT_trans = compute_SE_OT(X,Y,Q,R,g)

            # Classical OT cost
            C_trans = np.dot(C2, R)
            C_trans = np.dot(C1, C_trans)
            C_trans = C_trans / g
            G = np.dot(Q.T, C_trans)
            OT_trans = np.trace(G)

            if niter > 10:
                ## Update the error: theoritical error
                # err_1 = ((1/gamma)**2) * (KL(Q,Q_prev) + KL(Q_prev,Q))
                # err_2 = ((1/gamma)**2) * (KL(R,R_prev) + KL(R_prev,R))
                # err_3 = ((1/gamma)**2) * (KL(g,g_prev) + KL(g_prev,g))
                # err = err_1 + err_2 + err_3

                ## Update the error: Practical error
                err = np.abs(OT_trans - acc[-1]) / acc[-1]

                if np.isnan(err):
                    print("Error computation of the stopping criterion", niter)
                    Q = Q_prev
                    R = R_prev
                    g = g_prev
                    break

            if np.isnan(OT_trans) == True:
                print("Error: NaN OT value")
                return "Error"

            else:
                acc.append(OT_trans)
                end = time.time()
                tim_actual = end - start
                times.append(end - start)
                if tim_actual > time_out:
                    return (
                        acc[-1],
                        np.array(acc),
                        np.array(times),
                        Q,
                        R,
                        g,
                    )

        else:
            return (
                acc[-1],
                np.array(acc),
                np.array(times),
                Q,
                R,
                g,
            )

    return acc[-1], np.array(acc), np.array(times), Q, R, g


#################### Cost Matrix #####################


def Square_Euclidean_Distance(X, Y):
    """Returns the matrix of $|x_i-y_j|^2$."""
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sum((X_col - Y_lin) ** 2, 2)
    return C


# shape of xs: num_samples * dimension
def factorized_square_Euclidean(xs, xt):

    square_norm_s = np.sum(xs ** 2, axis=1)  # 2 * n * d
    square_norm_t = np.sum(xt ** 2, axis=1)  # 2 * m * d
    A_1 = np.zeros((np.shape(xs)[0], 2 + np.shape(xs)[1]))
    A_1[:, 0] = square_norm_s
    A_1[:, 1] = np.ones(np.shape(xs)[0])
    A_1[:, 2:] = -2 * xs  # n * d

    A_2 = np.zeros((2 + np.shape(xs)[1], np.shape(xt)[0]))
    A_2[0, :] = np.ones(np.shape(xt)[0])
    A_2[1, :] = square_norm_t
    A_2[2:, :] = xt.T

    return A_1, A_2


def Euclidean_Distance(X, Y):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sum((X_col - Y_lin) ** 2, 2)
    C = np.sqrt(C)
    return C


def Lp_Distance(X, Y, p=1):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sum(np.abs(X_col - Y_lin) ** p, 2)
    C = C ** (1 / p)
    return C


def Learning_linear_subspace(X, Y, cost, U, C_init=False, tol=1e-3):
    rank, m = np.shape(U)
    U_sym = np.dot(U, U.T)  # k x k
    d, v = np.linalg.eigh(U_sym)
    v = v / np.sqrt(d)  # k x k

    ind_column = np.random.choice(m, size=int(rank / tol))
    U_trans = U[:, ind_column]  # k x k/tol

    if C_init == False:
        A_trans = cost(X, Y[ind_column, :])
    else:
        A_trans = cost[:, ind_column]  # n x k/tol

    A_trans = (1 / np.sqrt(int(rank / tol))) * A_trans
    B = (1 / np.sqrt(int(rank / tol))) * np.dot(v.T, U_trans)  # k x k/tol
    Mat = np.linalg.inv(np.dot(B, B.T))
    Mat = np.dot(Mat, B)  # k x k/tol
    alpha = np.dot(Mat, A_trans.T)  # k x n

    V_f = np.dot(alpha.T, v.T)

    return V_f


# If C_init == True: cost is the Matrix
# If C_init == False: cost is the Function
def factorized_distance_cost(X, Y, rank, cost, C_init=False, tol=1e-3, seed=49):
    np.random.seed(seed)
    if C_init == False:
        n, m = np.shape(X)[0], np.shape(Y)[0]
    else:
        n, m = np.shape(cost)

    i_ = np.random.randint(n, size=1)
    j_ = np.random.randint(m, size=1)

    if C_init == False:
        X_trans = X[i_, :].reshape(1, -1)
        cost_trans_i = cost(X_trans, Y)
        mean = np.mean(cost_trans_i ** 2)
    else:
        cost_trans_i = cost[i_, :]
        mean = np.mean(cost_trans_i ** 2)

    if C_init == False:
        Y_trans = Y[j_, :].reshape(1, -1)
        cost_trans_j = cost(X, Y_trans)
    else:
        cost_trans_j = cost[:, j_]

    p_row = cost_trans_j ** 2 + cost_trans_i[0, j_] ** 2 + mean
    p_row = p_row / np.sum(p_row)  # vector of size n

    # Compute S
    ind_row = np.random.choice(n, size=int(rank / tol), p=p_row.reshape(-1))
    if C_init == False:
        S = cost(X[ind_row, :], Y)  # k/tol x m
    else:
        S = cost[ind_row, :]

    p_row_sub = p_row[ind_row]
    S = S / np.sqrt(int(rank / tol) * p_row_sub)

    norm_square_S = np.sum(S ** 2)
    p_column = np.zeros(m)
    for j in range(m):
        p_column[j] = np.sum(S[:, j] ** 2) / norm_square_S
    # p_column = p_column / np.sum(p_column) # vector of size m

    # Compute W
    ind_column = np.random.choice(m, size=int(rank / tol), p=p_column.reshape(-1))
    W = S[:, ind_column]  # k/tol x k/tol
    p_column_sub = p_column[ind_column]
    W = (W.T / np.sqrt(int(rank / tol) * p_column_sub)).T

    # Compute U
    u, d, v = np.linalg.svd(W)
    U = u[:, :rank]  # k/tol x k
    U_trans = np.dot(W.T, U)  # k/tol x k

    norm_U = np.sum(U_trans ** 2, axis=0)
    norm_U = np.sqrt(norm_U)

    U = np.dot(S.T, U)  # m x k
    U = U / norm_U

    # Compute V
    V = Learning_linear_subspace(X, Y, cost, U.T, C_init=C_init, tol=tol)

    return V, U.T


# Approximate the kernel k(x,y) = exp(TU/\varepsilon)
def RF_Approx(T, U, reg, num_samples=100, seed=49):
    R = np.minimum(theoritical_R(T, U.T), 100)
    A = Feature_Map_Gaussian(T, reg, R, num_samples=num_samples, seed=seed)
    B = Feature_Map_Gaussian(U.T, reg, R, num_samples=num_samples, seed=seed).T

    n, d = np.shape(T)
    m, d = np.shape(U.T)

    return A, B


def theoritical_R(X, Y):
    norm_X = np.linalg.norm(X, axis=1)  # n * d
    norm_Y = np.linalg.norm(Y, axis=1)  # m * d
    norm_max = np.maximum(np.max(norm_X), np.max(norm_Y))  # n + m

    return norm_max


## Feature map of k(x,y) = exp(\langle x,y\rangle/\varepsilon) ##
def Feature_Map_Gaussian(X, reg, R, num_samples=100, seed=49, tresh=700):
    n, d = np.shape(X)

    y = R ** 2 / (2 * reg * d)
    q = np.real((1 / 2) * np.exp(special.lambertw(y)))

    C = (2 * q) ** (d / 4)

    var = (q * reg) / 2

    np.random.seed(seed)
    U = np.random.multivariate_normal(np.zeros(d), var * np.eye(d), num_samples)
    SED = Square_Euclidean_Distance(X, U)  # d * n * num_samples
    W = -(SED) / reg  # n * num_samples
    V = np.sum(U ** 2, axis=1) / (2 * reg * q)  # num_samples * d + num_samples
    Z = np.sum(X ** 2, axis=1) / (2 * reg)  # n * d + n

    res_trans = V + W  # n * num_samples
    res_trans = np.minimum((Z + res_trans.T).T, tresh)  # 2 * n * num_samples
    res_trans = C * np.exp(res_trans)  # 2 * n * num_samples

    res = (1 / np.sqrt(num_samples)) * res_trans  # n * num_samples

    return res


# Uniform Random Nyström
# Here we approximate the SDP matrix exp(XX.T/\varepsilon)
def Nys_approx(X, Y, reg, rank, seed=49, stable=1e-10):
    n, d = np.shape(X)
    m, d = np.shape(Y)
    n_tot = n + m
    Z = np.concatenate((X, Y), axis=0)

    rank_trans = int(np.minimum(rank, n_tot))

    np.random.seed(seed)
    ind = np.random.choice(n_tot, rank_trans, replace=False)
    ind = np.sort(ind)

    Z_1 = Z[ind, :]
    A = np.exp(np.dot(Z_1, Z_1.T) / reg)
    A = A + stable * np.eye(rank_trans)
    V = np.exp(np.dot(Z, Z_1.T) / reg)

    return A, V
