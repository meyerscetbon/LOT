import numpy as np
import time
from sklearn.cluster import KMeans
import sklearn
import scipy
from scipy import special
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix


# Here C = C1 * C2 and P = P1 * P2
def compute_OT(P1, P2, C1, C2):
    OT_trans_1 = np.dot(P1.T, C1)
    OT_trans_2 = np.dot(C2, P2.T)
    OT_trans = np.dot(OT_trans_1, OT_trans_2)
    res = np.trace(OT_trans)
    return res


def compute_SE_OT(X, Y, Q, R, g):
    Q_trans = Q / np.sqrt(g)
    R_trans = R / np.sqrt(g)
    A = np.dot(X.T, Q_trans)
    B = np.dot(Y.T, R_trans)
    res = np.sum((A - B) ** 2)
    return res


def Sinkhorn(
    C, reg, a, b, C_init=False, max_iter=1000, delta=1e-3, lam=0, time_out=200
):
    start = time.time()
    acc = []
    times = []

    C = C / C.max()
    n, m = np.shape(a)[0], np.shape(b)[0]

    K = np.exp(-C / reg)
    # Next 3 lines equivalent to K= np.exp(-C/reg), but faster to compute
    # K = np.empty(C.shape, dtype=C.dtype)
    # np.divide(C, -reg, out=K)
    # np.exp(K, out=K)
    P = K.copy()
    v = np.ones(np.shape(b)[0])
    u_trans = np.dot(K, v) + lam  # add regularization to avoid divide 0

    OT_trans = np.sum(P * C)
    acc.append(OT_trans)
    time_actual = time.time() - start
    times.append(time_actual)

    err = 1
    n_iter = 0
    while (n_iter < max_iter) and (time_actual < time_out):
        P_prev = P
        if err > delta:
            n_iter = n_iter + 1

            # Update u
            u = a / u_trans

            # Update v
            v_trans = np.dot(K.T, u) + lam
            v = b / v_trans

            # Update the coupling
            P = u.reshape((-1, 1)) * K * v.reshape((1, -1))
            # Update the total cost
            OT_trans = np.sum(P * C)
            if np.isnan(OT_trans) == True:
                print("Error Sinkhorn: ", n_iter)
                P = P_prev
                break

            # Update the error
            u_trans = np.dot(K, v) + lam
            err = np.sum(np.abs(u * u_trans - a))
            # print(err)

            if np.isnan(err) == True:
                print("Error Sinkhorn: ", n_iter)
                P = P_prev
                break

            acc.append(OT_trans)
            time_actual = time.time() - start
            times.append(time_actual)

        else:
            break

    num_op = 3 * n * m + (n_iter + 1) * (2 * n * m + n + m)
    return acc[-1], np.array(acc), np.array(times), P, num_op


def Sinkhorn_LSE(C, reg, a, b, max_iter=1000, delta=1e-3, lam=0, time_out=200):
    start = time.time()
    acc = []
    times = []

    C = C / C.max()
    n, m = np.shape(a)[0], np.shape(b)[0]

    f = np.zeros(n)
    g = np.zeros(m)

    C_trans = -C / reg
    P = np.exp(C_trans)
    OT_trans = np.sum(P * C)
    acc.append(OT_trans)

    time_actual = time.time() - start
    times.append(time_actual)

    err = 1
    n_iter = 0
    while n_iter < max_iter and (time_actual < time_out):
        P_prev = P
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

            # Update the total cost
            OT_trans = np.sum(P * C)
            if np.isnan(OT_trans) == True:
                print("Error Sinkhorn: ", n_iter)
                P = P_prev
                break

            # Update the error
            err = np.sum(np.abs(np.sum(P, axis=1) - a))
            if np.isnan(err) == True:
                print("Error Sinkhorn: ", n_iter)
                P = P_prev
                break

            acc.append(OT_trans)
            time_actual = time.time() - start
            times.append(time_actual)

        else:
            break

    num_ops = 3 * n * m + (n_iter + 1) * (2 * n * m + n + m)
    return acc[-1], np.array(acc), np.array(times), P, num_ops


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
# Here we have assumed that to compute each entries of thecost matrix it takes O(d)
def UpdatePlans(X, Y, Z, a, b, reg, cost, max_iter=1000, delta=1e-9, lam=0):

    C1 = cost(Z, X)  # d * n * r
    C1 = C1 / C1.max()
    K1 = np.exp(-C1 / reg)  # size: r x n

    C2 = cost(Z, Y)  # d * m * r
    C2 = C2 / C2.max()
    K2 = np.exp(-C2 / reg)  # size: r x m

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
                print("Warning: numerical errors UpdatePlans at iteration", n_iter)
                u1, v1 = u1_prev, v1_prev
                u2, v2 = u2_prev, v2_prev
                w = w_prev
                break
        else:
            gamma_1 = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
            gamma_2 = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))
            n, m, d = np.shape(X)[0], np.shape(Y)[0], np.shape(Z)[1]
            count_op = (
                (n_iter + 1) * (2 * n * r + 2 * m * r + 6 * r + n + m)
                + (d + 2) * n * r
                + (d + 2) * m * r
                + r
            )
            return gamma_1, gamma_2, w, count_op

    gamma_1 = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
    gamma_2 = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))
    n, m, d = np.shape(X)[0], np.shape(Y)[0], np.shape(Z)[1]
    count_op = (
        (n_iter + 1) * (2 * n * r + 2 * m * r + 6 * r + n + m)
        + (d + 2) * n * r
        + (d + 2) * m * r
        + r
    )
    return gamma_1, gamma_2, w, count_op


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

            # Update the total cost

            # Metric used in the MIT paper
            # OT_trans = compute_SE_OT(X,Y,gamma1.T,gamma2.T,w)

            # Classical OT
            C_trans = np.dot(C, gamma2.T)
            C_trans = C_trans / w
            G = np.dot(gamma1, C_trans)
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


def LR_Dykstra(K1, K2, K3, gamma, a, b, alpha, max_iter=1000, delta=1e-9, lam=0):
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
                print("Error Dykstra: ", n_iter)
                u1, v1 = u1_prev, v1_prev
                u2, v2 = u2_prev, v2_prev
                g = g_prev
                break
        else:
            Q = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
            R = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))
            n, m = np.shape(K1)[0], np.shape(K2)[0]
            return Q, R, g, np.log(u1), np.log(v1), np.log(u2), np.log(v2)

    Q = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
    R = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))
    n, m = np.shape(K1)[0], np.shape(K2)[0]
    return (
        Q,
        R,
        g,
        np.log(u1) / gamma,
        np.log(v1) / gamma,
        np.log(u2) / gamma,
        np.log(v2) / gamma,
    )


# Approximate the kernel k(x,y) = exp(TU/\varepsilon)
def RF_Approx(T, U, reg, num_samples=100, seed=49):
    R = np.minimum(theoritical_R(T, U.T), 100)
    A = Feature_Map_Gaussian(T, reg, R, num_samples=num_samples, seed=seed)
    B = Feature_Map_Gaussian(U.T, reg, R, num_samples=num_samples, seed=seed).T

    n, d = np.shape(T)
    m, d = np.shape(U.T)

    num_op = (
        d * n * num_samples
        + 6 * n * num_samples
        + num_samples * d
        + n * d
        + num_samples
        + n
    )
    num_op = (
        num_op
        + d * m * num_samples
        + 6 * m * num_samples
        + num_samples * d
        + m * d
        + num_samples
        + m
    )
    num_op = num_op + n * d + m * d + n + m

    return A, B, num_op


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


#################### Cost Matrix #####################
## Feature map of k(x,y) = \langle x,y\rangle ** 2 ##
def Feature_Map_Poly(X):
    n, d = np.shape(X)
    X_new = np.zeros((n, d**2))
    for i in range(n):
        x = X[i, :][:, None]
        X_new[i, :] = np.dot(x, x.T).reshape(-1)
    return X_new


def theoritical_R(X, Y):
    norm_X = np.linalg.norm(X, axis=1)
    norm_Y = np.linalg.norm(Y, axis=1)
    norm_max = np.maximum(np.max(norm_X), np.max(norm_Y))

    return norm_max


### Random Feature Maps of RBF Kernel
def Feature_Map_Gaussian(X, reg, R=1, num_samples=100, seed=49):
    n, d = np.shape(X)

    y = R**2 / (reg * d)
    q = np.real((1 / 2) * np.exp(special.lambertw(y)))
    C = (2 * q) ** (d / 4)

    var = (q * reg) / 4

    np.random.seed(seed)
    U = np.random.multivariate_normal(np.zeros(d), var * np.eye(d), num_samples)

    SED = Square_Euclidean_Distance(X, U)
    W = -(2 * SED) / reg
    V = np.sum(U**2, axis=1) / (reg * q)

    res_trans = V + W
    res_trans = C * np.exp(res_trans)

    res = (1 / np.sqrt(num_samples)) * res_trans

    return res


def Square_Euclidean_Distance(X, Y):
    """Returns the matrix of $|x_i-y_j|^2$."""
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sum((X_col - Y_lin) ** 2, 2)
    # D = (np.sum(X ** 2, 1)[:, np.newaxis] - 2 * np.dot(X, Y.T) + np.sum(Y ** 2, 1))
    return C


# shape of xs: num_samples * dimension
def factorized_square_Euclidean(xs, xt):

    square_norm_s = np.sum(xs**2, axis=1)  # 2 * n * d
    square_norm_t = np.sum(xt**2, axis=1)  # 2 * m * d
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
    # D = (np.sum(X ** 2, 1)[:, np.newaxis] - 2 * np.dot(X, Y.T) + np.sum(Y ** 2, 1))
    return C


def Lp_Distance(X, Y, p=1):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sum(np.abs(X_col - Y_lin) ** p, 2)
    C = C ** (1 / p)
    # D = (np.sum(X ** 2, 1)[:, np.newaxis] - 2 * np.dot(X, Y.T) + np.sum(Y ** 2, 1))
    return C


def rbf_distance(X):
    kernel = sklearn.metrics.pairwise.rbf_kernel(X)
    D = 1 - kernel
    return D


def Learning_linear_subspace(X, Y, cost, U, C_init=False, tol=1e-3):
    rank, m = np.shape(U)
    U_sym = np.dot(U, U.T)  # k x k
    # d, v = np.linalg.eigh(U_sym)
    u, d, v_transpose = np.linalg.svd(U_sym)
    v = v_transpose.T
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
        X_trans = X[i_, :]
        if np.shape(X_trans)[0] != 1:
            X_trans = X_trans[np.newaxis, :]
        cost_trans_i = cost(X_trans, Y)
        mean = np.mean(cost_trans_i**2)
    else:
        cost_trans_i = cost[i_, :]
        mean = np.mean(cost_trans_i**2)

    if C_init == False:
        Y_trans = Y[j_, :]
        if np.shape(Y_trans)[0] != 1:
            Y_trans = Y_trans[np.newaxis, :]
        cost_trans_j = cost(X, Y_trans)
    else:
        cost_trans_j = cost[:, j_]

    p_row = cost_trans_j**2 + cost_trans_i[0, j_] ** 2 + mean
    p_row = p_row / np.sum(p_row)  # vector of size n

    # Compute S
    ind_row = np.random.choice(n, size=int(rank / tol), p=p_row.reshape(-1))
    if C_init == False:
        S = cost(X[ind_row, :], Y)  # k/tol x m
    else:
        S = cost[ind_row, :]

    p_row_sub = p_row[ind_row]
    S = S / np.sqrt(int(rank / tol) * p_row_sub)

    norm_square_S = np.sum(S**2)
    p_column = np.zeros(m)
    for j in range(m):
        p_column[j] = np.sum(S[:, j] ** 2) / norm_square_S

    p_column = p_column / np.sum(p_column)  # vector of size m
    # Compute W
    ind_column = np.random.choice(m, size=int(rank / tol), p=p_column.reshape(-1))
    W = S[:, ind_column]  # k/tol x k/tol
    p_column_sub = p_column[ind_column]
    W = (W.T / np.sqrt(int(rank / tol) * p_column_sub)).T

    # Compute U
    u, d, v = np.linalg.svd(W)
    U = u[:, :rank]  # k/tol x k
    U_trans = np.dot(W.T, U)  # k/tol x k

    norm_U = np.sum(U_trans**2, axis=0)
    norm_U = np.sqrt(norm_U)

    U = np.dot(S.T, U)  # m x k
    U = U / norm_U

    # Compute V
    V = Learning_linear_subspace(X, Y, cost, U.T, C_init=C_init, tol=tol)

    return V, U.T


# compute the connectivity matrix of a distance matrix
def k_smallest_by_row(D, k=50):
    ind_D = np.argpartition(D, k)
    ind_D_trans = ind_D[:, :k]
    row_indices = tuple(
        np.full(len(col_index), i) for i, col_index in enumerate(ind_D_trans)
    )
    row_indices = np.concatenate(row_indices)
    col_indices = np.concatenate(ind_D_trans)
    mask = np.zeros((np.shape(D)[0], np.shape(D)[1]))
    mask[row_indices, col_indices] = 1

    return mask


## shortest path distance for graphs
def shortest_path_distance(X, graph_type="kneighbors_graph", n_neighbors=10):
    if graph_type == "kneighbors_graph":
        csr_graph = sklearn.neighbors.kneighbors_graph(X, n_neighbors, mode="distance")
    if graph_type == "rbf":
        kernel = sklearn.metrics.pairwise.rbf_kernel(X)
        csr_graph = 1 - kernel
    D = dijkstra(csr_graph, directed=False, return_predecessors=False, unweighted=False)
    return D


######## Factorized shortest path distance matrix for graphs
def factorised_shortest_path_distance_kernel(
    X, num_connection, rank_rf=100, rank=100, tol=1e-3, seed=49
):
    reg = np.shape(X)[1]
    R = theoritical_R(X, X)
    phi_X = Feature_Map_Gaussian(X, reg, R=R, num_samples=rank_rf, seed=seed)
    kernel = np.dot(phi_X, phi_X.T)
    rescale = np.max(kernel)
    csr_graph = rescale - kernel
    csr_graph = k_smallest_by_row(csr_graph, k=num_connection)
    D = factorized_shortest_path(csr_graph, rank, tol=tol, seed=seed + 10)
    return D


## Here k is the number of connectivity allowed
## Here rank_cost is the rank of the factorization of the distance matrix
## Here the cost must be a metric as we factorize it to compute the graph
def k_connectivity_graph(data, k, cost, rank_cost=100, seed=49):
    cost_factorized = lambda X, Y: factorized_distance_cost(
        X, Y, rank_cost, cost, C_init=False, tol=1e-1, seed=seed
    )
    D11, D12 = cost_factorized(data, data)
    D = np.dot(D11, D12)
    graph_data = k_smallest_by_row(D, k=k)
    csr_graph = csr_matrix(graph_data)

    return csr_graph


## here csr_graph is the sparse connectivity graph
# G = dijkstra(G_trans, directed=False, indices=[], return_predecessors=False, unweighted=False)
def Learning_linear_subspace_shortest_path(csr_graph, U, tol=1e-3):
    rank, m = np.shape(U)
    U_sym = np.dot(U, U.T)  # k x k
    # d, v = np.linalg.eigh(U_sym)
    u, d, v_transpose = np.linalg.svd(U_sym)
    v = v_transpose.T
    v = v / np.sqrt(d)  # k x k

    ind_column = np.random.choice(m, size=int(rank / tol))
    U_trans = U[:, ind_column]  # k x k/tol

    A_trans = dijkstra(
        csr_graph,
        directed=False,
        indices=ind_column,
        return_predecessors=False,
        unweighted=False,
    )
    A_trans = A_trans.T

    A_trans = (1 / np.sqrt(int(rank / tol))) * A_trans
    B = (1 / np.sqrt(int(rank / tol))) * np.dot(v.T, U_trans)  # k x k/tol
    Mat = np.linalg.inv(np.dot(B, B.T))
    Mat = np.dot(Mat, B)  # k x k/tol
    alpha = np.dot(Mat, A_trans.T)  # k x n

    V_f = np.dot(alpha.T, v.T)

    return V_f


## here csr_graph is the sparse connectivity graph
def factorized_shortest_path(csr_graph, rank, tol=1e-3, seed=49):
    np.random.seed(seed)

    n, m = np.shape(csr_graph)[0], np.shape(csr_graph)[1]
    i_ = np.random.randint(n, size=1)
    j_ = np.random.randint(m, size=1)

    cost_trans_i = dijkstra(
        csr_graph,
        directed=False,
        indices=i_,
        return_predecessors=False,
        unweighted=False,
    )
    cost_trans_i = cost_trans_i.reshape(-1)
    mean = np.mean(cost_trans_i**2)
    cost_trans_j = dijkstra(
        csr_graph,
        directed=False,
        indices=j_,
        return_predecessors=False,
        unweighted=False,
    )
    cost_trans_j = cost_trans_j.reshape(-1)

    p_row = cost_trans_j**2 + cost_trans_i[j_] ** 2 + mean
    p_row = p_row / np.sum(p_row)  # probability of size n

    # Compute S
    ind_row = np.random.choice(n, size=int(rank / tol), p=p_row.reshape(-1))
    S = dijkstra(
        csr_graph,
        directed=False,
        indices=ind_row,
        return_predecessors=False,
        unweighted=False,
    )

    p_row_sub = p_row[ind_row]
    S = (S.T / np.sqrt(int(rank / tol) * p_row_sub)).T

    norm_square_S = np.sum(S**2)
    p_column = np.zeros(m)
    for j in range(m):
        p_column[j] = np.sum(S[:, j] ** 2) / norm_square_S

    p_column = p_column / np.sum(p_column)  # vector of size m
    # Compute W
    ind_column = np.random.choice(m, size=int(rank / tol), p=p_column.reshape(-1))
    W = S[:, ind_column]  # k/tol x k/tol
    p_column_sub = p_column[ind_column]
    W = (W.T / np.sqrt(int(rank / tol) * p_column_sub)).T

    # Compute U
    u, d, v = np.linalg.svd(W)
    U = u[:, :rank]  # k/tol x k
    U_trans = np.dot(W.T, U)  # k/tol x k

    norm_U = np.sum(U_trans**2, axis=0)
    norm_U = np.sqrt(norm_U)

    U = np.dot(S.T, U)  # m x k
    U = U / norm_U

    # Compute V
    V = Learning_linear_subspace_shortest_path(csr_graph, U.T, tol=tol)

    return V, U.T
