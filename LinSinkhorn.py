import utils
import numpy as np
import time
from sklearn.cluster import KMeans
import scipy


def KL(A, B):
    Ratio_trans = np.log(A) - np.log(B)
    return np.sum(A * Ratio_trans)


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
                print("Warning: numerical error in Dykstra at iteration: ", n_iter)
                u1, v1 = u1_prev, v1_prev
                u2, v2 = u2_prev, v2_prev
                g = g_prev
                break
        else:
            Q = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
            R = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))
            n, m = np.shape(K1)[0], np.shape(K2)[0]
            count_op = (
                (n_iter + 1) * (20 * r + 2 * n * r + 2 * m * r + n + m)
                + 2 * n * r
                + 2 * m * r
            )
            return Q, R, g, count_op, n_iter

    Q = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
    R = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))
    n, m = np.shape(K1)[0], np.shape(K2)[0]
    count_op = (
        (n_iter + 1) * (20 * r + 2 * n * r + 2 * m * r + n + m) + 2 * n * r + 2 * m * r
    )
    return Q, R, g, count_op, n_iter


def LR_Dykstra_LSE_Sin(
    C1, C2, C3, a, b, alpha, gamma, max_iter=1000, delta=1e-9, lam=0
):

    h_old = -C3
    r = np.shape(C3)[0]
    g1_old, g2_old = np.zeros(r), np.zeros(r)
    f1, f2 = np.zeros(np.shape(a)[0]), np.zeros(np.shape(b)[0])

    w_gi, w_gp = np.zeros(r), np.zeros(
        r
    )  # q_gi, q_gp = np.exp(gamma * w_gi), np.exp(gamma * w_gp)
    w_Q, w_R = np.zeros(r), np.zeros(
        r
    )  # q_Q, q_R = np.exp(gamma * w_Q), np.exp(gamma * w_R)

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

            h = w_gi + h_old  # 2 * r
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
                print("Warning: numerical error in Dykstra LSE at iteration", n_iter)
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

                n, m = np.shape(C1)[0], np.shape(C2)[0]
                count_op = (
                    (n_iter) * (8 * n * r + 8 * m * r + 4 * n + 4 * m + 27 * r)
                    + 4 * n * r
                    + 4 * m * r
                )
                return Q, R, g, count_op

        else:
            n, m = np.shape(C1)[0], np.shape(C2)[0]
            count_op = (
                (n_iter + 1) * (8 * n * r + 8 * m * r + 4 * n + 4 * m + 27 * r)
                + 4 * n * r
                + 4 * m * r
            )
            return Q, R, g, count_op

    n, m = np.shape(C1)[0], np.shape(C2)[0]
    count_op = (
        (n_iter + 1) * (8 * n * r + 8 * m * r + 4 * n + 4 * m + 27 * r)
        + 4 * n * r
        + 4 * m * r
    )
    return Q, R, g, count_op


def LR_IBP_Sin(K1, K2, K3, a, b, max_iter=1000, delta=1e-9, lam=0):
    Q = K1
    R = K2
    g = K3

    r = np.shape(K3)[0]
    v1, v2 = np.ones(r), np.ones(r)
    u1, u2 = np.ones(np.shape(a)[0]), np.ones(np.shape(b)[0])

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
            u2 = b / u2_trans  # m
            v2_trans = np.dot(K2.T, u2)  # m * r

            # Update g
            # g = g / np.sum(g)
            g = (g * v1 * v1_trans * v2 * v2_trans) ** (1 / 3)  # 5 * r

            # Update v1
            v1 = g / v1_trans  # r

            # Update v2
            v2 = g / v2_trans  # r

            # Update the couplings
            # Q = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
            # R = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))

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
                print("Warning: numerical errors in IBP at iteration", n_iter)
                u1, v1 = u1_prev, v1_prev
                u2, v2 = u2_prev, v2_prev
                g = g_prev
                break
        else:
            Q = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
            R = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))
            n, m = np.shape(K1)[0], np.shape(K2)[0]
            count_op = (
                (n_iter + 1) * (2 * n * r + 2 * m * r + 7 * r) + 3 * n * r + 3 * m * r
            )
            return Q, R, g, count_op

    Q = u1.reshape((-1, 1)) * K1 * v1.reshape((1, -1))
    R = u2.reshape((-1, 1)) * K2 * v2.reshape((1, -1))
    n, m = np.shape(K1)[0], np.shape(K2)[0]
    count_op = (n_iter + 1) * (2 * n * r + 2 * m * r + 7 * r) + 3 * n * r + 3 * m * r
    return Q, R, g, count_op


def self_quad_lot_md(
    C,
    a,
    rank,
    gamma_0=1,
    LSE="False",
    alpha=1e-10,
    seed_init=49,
    max_iter=1000,
    delta=1e-5,
    max_iter_Sin=10000,
    delta_Sin=1e-3,
    lam_Sin=0,
    time_out=200,
):
    start = time.time()
    num_op = 0
    acc = []
    times = []
    list_num_op = []

    n = np.shape(a)[0]
    rank = min(rank, n)
    r = rank

    # rescale the cost
    C = C / C.max()

    # Init g
    g = np.ones(rank) / rank

    # Init Q
    np.random.seed(seed_init)
    Q = np.abs(np.random.randn(n, rank))
    Q = Q + 1  # n * r
    Q = (Q.T * (a / np.sum(Q, axis=1))).T  # n + n * r

    # Classical OT
    C_trans = np.dot(C, Q)
    C_trans = C_trans / g
    G = np.dot(Q.T, C_trans)
    OT_trans = np.trace(G)

    acc.append(OT_trans)
    num_op = num_op + n * r + n * n * r + r * r * n
    list_num_op.append(num_op)
    time_actual = time.time() - start
    times.append(time_actual)

    err = 1
    niter = 0
    count_escape = 1
    while (niter < max_iter) and (time_actual < time_out):
        Q_prev = Q
        g_prev = g
        if err > delta:
            niter = niter + 1

            grad = np.dot(C, Q) + np.dot(C.T, Q)
            grad = grad / g
            norm = np.max(np.abs(grad)) ** 2
            gamma = gamma_0 / norm

            C_trans = grad - (1 / gamma) * np.log(Q)  # 3 * n * r
            num_op = num_op + 2 * n * n * r + 2 * n * r

            # Sinkhorn
            reg = 1 / gamma
            if LSE == "False":
                results = utils.Sinkhorn(
                    C_trans,
                    reg,
                    a,
                    g,
                    max_iter=max_iter_Sin,
                    delta=delta_Sin,
                    lam=lam_Sin,
                    time_out=time_out,
                )

            else:
                results = utils.Sinkhorn_LSE(
                    C_trans,
                    reg,
                    a,
                    g,
                    max_iter=max_iter_Sin,
                    delta=delta_Sin,
                    lam=lam_Sin,
                    time_out=time_out,
                )

            res_sin, acc_sin, times_sin, Q, num_op_sin = results
            num_op = num_op + num_op_sin

            # Classical OT
            C_trans = np.dot(C, Q)
            C_trans = C_trans / g
            G = np.dot(Q.T, C_trans)
            OT_trans = np.trace(G)

            if np.isnan(OT_trans) == True:
                print("Error self LOT: OT cost", niter)
                Q = Q_prev
                g = g_prev
                break

            ## Update the error: theoritical error
            criterion = ((1 / gamma) ** 2) * (KL(Q, Q_prev) + KL(Q_prev, Q))
            if niter > 1:
                if criterion > delta / 1e-1:
                    err = criterion
                else:
                    count_escape = count_escape + 1
                    if count_escape != niter:
                        err = criterion

            ## Update the error: Practical error
            # err = np.abs(OT_trans - acc[-1]) / acc[-1]

            if np.isnan(criterion):
                print("Error self LOT: stopping criterion", niter)
                Q = Q_prev
                g = g_prev
                break

            acc.append(OT_trans)
            list_num_op.append(num_op)
            time_actual = time.time() - start
            times.append(time_actual)

        else:
            break

    return acc[-1], np.array(acc), np.array(times), np.array(list_num_op), Q


def self_lin_lot_md(
    C1,
    C2,
    a,
    rank,
    gamma_0=1,
    LSE="True",
    alpha=1e-10,
    seed_init=49,
    max_iter=1000,
    delta=1e-3,
    max_iter_Sin=1000,
    delta_Sin=1e-9,
    lam_Sin=0,
    time_out=200,
):
    start = time.time()
    num_op = 0
    acc = []
    times = []
    list_num_op = []

    n, d = np.shape(C1)
    rank = min(n, rank)
    r = rank

    # rescale the costs
    C1 = C1 / np.sqrt(C1.max())
    C2 = C2 / np.sqrt(C2.max())

    # Init g
    g = np.ones(rank) / rank

    # Init Q
    np.random.seed(seed_init)
    Q = np.abs(np.random.randn(n, rank))
    Q = Q + 1  # n * r
    Q = (Q.T * (a / np.sum(Q, axis=1))).T  # n + n * r

    # Classical OT
    C_trans = np.dot(C2, Q)
    C_trans = np.dot(C1, C_trans)
    C_trans = C_trans / g
    G = np.dot(Q.T, C_trans)
    OT_trans = np.trace(G)

    acc.append(OT_trans)
    num_op = num_op + 3 * n * r + n + r
    list_num_op.append(num_op)
    time_actual = time.time() - start
    times.append(time_actual)

    err = 1
    niter = 0
    count_escape = 1
    while (niter < max_iter) and (time_actual < time_out):
        Q_prev = Q
        g_prev = g
        if err > delta:
            niter = niter + 1

            grad = np.dot(C1, np.dot(C2, Q)) + np.dot(C2.T, np.dot(C1.T, Q))
            grad = grad / g
            norm = np.max(np.abs(grad)) ** 2
            gamma = gamma_0 / norm

            C_trans = grad - (1 / gamma) * np.log(Q)  # 3 * n * r

            num_op = num_op + 4 * n * d * r + 4 * n * r

            # Sinkhorn
            reg = 1 / gamma
            if LSE == "False":
                results = utils.Sinkhorn(
                    C_trans,
                    reg,
                    a,
                    g,
                    max_iter=max_iter_Sin,
                    delta=delta_Sin,
                    lam=lam_Sin,
                    time_out=time_out,
                )

            else:
                results = utils.Sinkhorn_LSE(
                    C_trans,
                    reg,
                    a,
                    g,
                    max_iter=max_iter_Sin,
                    delta=delta_Sin,
                    lam=lam_Sin,
                    time_out=time_out,
                )

            res_sin, acc_sin, times_sin, Q, num_op_sin = results
            num_op = num_op + num_op_sin

            # Classical OT
            C_trans = np.dot(C2, Q)
            C_trans = np.dot(C1, C_trans)
            C_trans = C_trans / g
            G = np.dot(Q.T, C_trans)
            OT_trans = np.trace(G)

            if np.isnan(OT_trans) == True:
                print("Error self LOT: OT cost", niter)
                Q = Q_prev
                g = g_prev
                break

            ## Update the error: theoritical error
            criterion = ((1 / gamma) ** 2) * (KL(Q, Q_prev) + KL(Q_prev, Q))
            if niter > 1:
                if criterion > delta / 1e-1:
                    err = criterion
                else:
                    count_escape = count_escape + 1
                    if count_escape != niter:
                        err = criterion

            ## Update the error: Practical error
            # err = np.abs(OT_trans - acc[-1]) / acc[-1]

            if np.isnan(err):
                print("Error self LOT: stopping criterion", niter)
                Q = Q_prev
                g = g_prev
                break

            acc.append(OT_trans)
            list_num_op.append(num_op)
            time_actual = time.time() - start
            times.append(time_actual)

        else:
            break

    return acc[-1], np.array(acc), np.array(times), np.array(list_num_op), Q


# If C_init == True: cost is the tuples C(X,Y), C(X,X), C(Y,Y)
# If C_init == False: cost is the Function
# Init == 'trivial', 'random', 'kmeans','general_kmeans'
def Quad_LOT_MD(
    X,
    Y,
    a,
    b,
    rank,
    cost,
    reg=0,
    alpha=1e-10,
    gamma_0=1,
    max_iter=100,
    delta=1e-5,
    time_out=100,
    Init="kmeans",
    seed_init=49,
    C_init=False,
    reg_init=1e-1,
    gamma_init="rescale",
    method="Dykstra",
    max_iter_IBP=10000,
    delta_IBP=1e-3,
    lam_IBP=0,
):
    num_op = 0
    acc = []
    times = []
    list_num_op = []

    if gamma_0 * reg >= 1:
        # display(Latex(f'Choose $\gamma$ and $\epsilon$ such that $\gamma$ x $\epsilon<1$'))
        print("gamma et epsilon must be well choosen")
        return "Error"

    n, m = np.shape(a)[0], np.shape(b)[0]
    rank = min(n, m, rank)
    r = rank

    if C_init == False:
        C = cost(X, Y)
        C = C / np.max(C)
        C_X = cost(X, X)
        C_X = C_X / C_X.max()
        C_Y = cost(Y, Y)
        C_Y = C_Y / C_Y.max()
    else:
        C, C_X, C_Y = cost
        C, C_X, C_Y = C / C.max(), C_X / C_X.max(), C_Y / C_Y.max()

    if len(C) == 2:
        print("Error: cost not adapted")
        return "Error"

    start = time.time()

    #### Initialization #####
    if Init == "general_kmeans":
        g = np.ones(rank) / rank
        res_q, acc_q, times_q, list_num_op_q, Q = self_quad_lot_md(
            C_X,
            a,
            rank,
            gamma_0=gamma_0,
            LSE=False,
            alpha=1e-10,
            seed_init=49,
            max_iter=10,
            delta=delta,
            max_iter_Sin=max_iter_IBP,
            delta_Sin=delta_IBP,
            lam_Sin=lam_IBP,
            time_out=time_out / 5,
        )
        res_r, acc_r, times_r, list_num_op_r, R = self_quad_lot_md(
            C_Y,
            b,
            rank,
            gamma_0=gamma_0,
            LSE=False,
            alpha=1e-10,
            seed_init=49,
            max_iter=10,
            delta=delta,
            max_iter_Sin=max_iter_IBP,
            delta_Sin=delta_IBP,
            lam_Sin=lam_IBP,
            time_out=time_out / 5,
        )

        num_op = num_op + list_num_op_q[-1] + list_num_op_r[-1]
    if Init == "kmeans":
        ## Init with K-means
        g = np.ones(rank) / rank
        kmeans = KMeans(n_clusters=rank, random_state=0).fit(X)
        Z = kmeans.cluster_centers_
        num_iter_kmeans = kmeans.n_iter_
        num_op = num_op + num_iter_kmeans * rank * np.shape(X)[0] + rank
        reg_init = reg_init
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
        num_op = num_op + count_op_Barycenter

    # Init random
    if Init == "random":
        np.random.seed(seed_init)
        g = np.abs(np.random.randn(rank))
        g = g + 1  # r
        g = g / np.sum(g)  # r

        Q = np.abs(np.random.randn(n, rank))
        Q = Q + 1  # n * r
        Q = (Q.T * (a / np.sum(Q, axis=1))).T  # n + n * r

        R = np.abs(np.random.randn(m, rank))
        R = R + 1  # m * r
        R = (R.T * (b / np.sum(R, axis=1))).T  # m + m * r
        num_op = num_op + 2 * n * r + 2 * m * r + m + n + 2 * r

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

        num_op = num_op + 4 * n * r + 4 * m * r + 3 * n + 3 * m + 3 * r

    if gamma_init == "theory":
        L_trans = (2 / (alpha) ** 4) * (np.linalg.norm(C) ** 2)
        L_trans = L_trans + ((reg + 2 * np.linalg.norm(C)) / (alpha**3)) ** 2
        L = np.sqrt(3 * L_trans)
        gamma = 1 / L

    if gamma_init == "regularization":
        gamma = 1 / reg

    if gamma_init == "arbitrary":
        gamma = gamma_0

    # Classical OT
    C_trans = np.dot(C, R)
    C_trans = C_trans / g
    G = np.dot(Q.T, C_trans)
    OT_trans = np.trace(G)

    acc.append(OT_trans)
    list_num_op.append(num_op)
    time_actual = time.time() - start
    times.append(time_actual)

    err = 1
    niter = 0
    count_escape = 1
    while (niter < max_iter) and (time_actual < time_out):
        Q_prev = Q
        R_prev = R
        g_prev = g
        if err > delta:
            niter = niter + 1

            K1_trans_0 = np.dot(C, R)  # n * m * r
            grad_Q = K1_trans_0 / g
            if reg != 0:
                grad_Q = grad_Q + reg * np.log(Q)
            if gamma_init == "rescale":
                # norm_1 = np.linalg.norm(grad_Q)**2
                norm_1 = np.max(np.abs(grad_Q)) ** 2

            K2_trans_0 = np.dot(C.T, Q)  # m * n * r
            grad_R = K2_trans_0 / g
            if reg != 0:
                grad_R = grad_R + reg * np.log(R)
            if gamma_init == "rescale":
                # norm_2 = np.linalg.norm(grad_R)**2
                norm_2 = np.max(np.abs(grad_R)) ** 2

            omega = np.diag(np.dot(Q.T, K1_trans_0))  # r * n * r
            C3_trans = omega / (g**2)
            grad_g = -omega / (g**2)
            if reg != 0:
                grad_g = grad_g + reg * np.log(g)

            if gamma_init == "rescale":
                # norm_3 = np.linalg.norm(grad_g)**2
                norm_3 = np.max(np.abs(grad_g)) ** 2

            if gamma_init == "rescale":
                gamma = gamma_0 / max(norm_1, norm_2, norm_3)

            C1_trans = grad_Q - (1 / gamma) * np.log(Q)  # 3 * n * r
            C2_trans = grad_R - (1 / gamma) * np.log(R)  # 3 * m * r
            C3_trans = grad_g - (1 / gamma) * np.log(g)  # 4 * r

            num_op = num_op + 2 * n * m * r + r * n * r + 3 * n * r + 3 * m * r + 4 * r

            # Update the coupling
            if method == "IBP":
                K1 = np.exp((-gamma) * C1_trans)
                K2 = np.exp((-gamma) * C2_trans)
                K3 = np.exp((-gamma) * C3_trans)
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
                K3 = np.exp((-gamma) * C3_trans)
                num_op = num_op + 2 * n * r + 2 * m * r + 2 * r
                Q, R, g, count_op_Dysktra, n_iter_Dykstra = LR_Dykstra_Sin(
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

                num_op = num_op + count_op_Dysktra

            if method == "Dykstra_LSE":
                Q, R, g, count_op_Dysktra_LSE = LR_Dykstra_LSE_Sin(
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

                num_op = num_op + count_op_Dysktra_LSE

            # Classical OT
            C_trans = np.dot(C, R)
            C_trans = C_trans / g
            G = np.dot(Q.T, C_trans)
            OT_trans = np.trace(G)

            if np.isnan(OT_trans) == True:
                print("Error LOT: OT cost", niter)
                Q = Q_prev
                R = R_prev
                g = g_prev
                break

            ## Update the error: theoritical error
            err_1 = ((1 / gamma) ** 2) * (KL(Q, Q_prev) + KL(Q_prev, Q))
            err_2 = ((1 / gamma) ** 2) * (KL(R, R_prev) + KL(R_prev, R))
            err_3 = ((1 / gamma) ** 2) * (KL(g, g_prev) + KL(g_prev, g))
            criterion = err_1 + err_2 + err_3
            # print(criterion)
            if niter > 1:
                if criterion > delta / 1e-1:
                    err = criterion
                else:
                    count_escape = count_escape + 1
                    if count_escape != niter:
                        err = criterion

            ## Update the error: Practical error
            # err = np.abs(OT_trans - acc[-1]) / acc[-1]

            if np.isnan(criterion) == True:
                print("Error LOT: stopping criterion", niter)
                Q = Q_prev
                R = R_prev
                g = g_prev
                break

            acc.append(OT_trans)
            list_num_op.append(num_op)
            time_actual = time.time() - start
            times.append(time_actual)

        else:
            break

    return acc[-1], np.array(acc), np.array(times), np.array(list_num_op), Q, R, g


# If C_init = True: cost_factorized = (C1, C2, C_X_1, C_X_2, C_Y_1, C_Y_2)
# If C_init = False: cost_factorized is a function
# Init = 'trivial', 'random', 'kmeans', 'general_kmeans'
def Lin_LOT_MD(
    X,
    Y,
    a,
    b,
    rank,
    cost,
    cost_factorized,
    reg=0,
    alpha=1e-10,
    gamma_0=10,
    max_iter=1000,
    delta=1e-3,
    time_out=200,
    Init="kmeans",
    seed_init=49,
    C_init=False,
    reg_init=1e-1,
    gamma_init="rescale",
    method="Dykstra",
    max_iter_IBP=10000,
    delta_IBP=1e-3,
    lam_IBP=0,
    rescale_cost=True,
):
    num_op = 0
    acc = []
    times = []
    list_num_op = []
    list_criterion = []

    if gamma_0 * reg >= 1:
        # display(Latex(f'Choose $\gamma$ and $\epsilon$ such that $\gamma$ x $\epsilon<1$'))
        print("gamma and epsilon must be well choosen")
        return "Error"

    n, m = np.shape(a)[0], np.shape(b)[0]
    rank = min(n, m, rank)
    r = rank

    if C_init == False:
        C = cost_factorized(X, Y)
        if len(C) == 2:
            C1, C2 = C
            if rescale_cost == True:
                C1, C2 = C1 / np.sqrt(np.max(C1)), C2 / np.sqrt(np.max(C2))

        C_X = cost_factorized(X, X)
        if len(C) == 2:
            C_X_1, C_X_2 = C_X
            if rescale_cost == True:
                C_X_1, C_X_2 = C_X_1 / np.sqrt(np.max(C_X_1)), C_X_2 / np.sqrt(
                    np.max(C_X_2)
                )

        C_Y = cost_factorized(Y, Y)
        if len(C) == 2:
            C_Y_1, C_Y_2 = C_Y
            if rescale_cost == True:
                C_Y_1, C_Y_2 = C_Y_1 / np.sqrt(np.max(C_Y_1)), C_Y_2 / np.sqrt(
                    np.max(C_Y_2)
                )
    else:
        (C1, C2, C_X_1, C_X_2, C_Y_1, C_Y_2) = cost_factorized

    n, d = np.shape(C1)
    start = time.time()

    ########### Initialization ###########
    #### Initialization #####
    if Init == "general_kmeans":
        g = np.ones(rank) / rank
        res_q, acc_q, times_q, list_num_op_q, Q = self_lin_lot_md(
            C_X_1,
            C_X_2,
            a,
            rank,
            gamma_0=gamma_0,
            LSE=False,
            seed_init=49,
            max_iter=10,
            alpha=alpha,
            delta=delta,
            max_iter_Sin=max_iter_IBP,
            delta_Sin=delta_IBP,
            lam_Sin=lam_IBP,
            time_out=1e100,
        )
        res_r, acc_r, times_r, list_num_op_r, R = self_lin_lot_md(
            C_Y_1,
            C_Y_2,
            b,
            rank,
            gamma_0=gamma_0,
            LSE=False,
            seed_init=49,
            max_iter=10,
            alpha=alpha,
            delta=delta,
            max_iter_Sin=max_iter_IBP,
            delta_Sin=delta_IBP,
            lam_Sin=lam_IBP,
            time_out=1e100,
        )
        num_op = num_op + list_num_op_q[-1] + list_num_op_r[-1]

    ## Init with K-means
    if Init == "kmeans":
        g = np.ones(rank) / rank
        kmeans = KMeans(n_clusters=rank, random_state=0).fit(X)
        Z = kmeans.cluster_centers_
        num_iter_kmeans = kmeans.n_iter_
        num_op = num_op + r + num_iter_kmeans * r * n
        reg_init = reg_init
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
        num_op = num_op + count_op_Barycenter

    ## Init random
    if Init == "random":
        np.random.seed(seed_init)
        g = np.abs(np.random.randn(rank))
        g = g + 1
        g = g / np.sum(g)
        n, d = np.shape(X)
        m, d = np.shape(Y)

        Q = np.abs(np.random.randn(n, rank))
        Q = Q + 1
        Q = (Q.T * (a / np.sum(Q, axis=1))).T

        R = np.abs(np.random.randn(m, rank))
        R = R + 1
        R = (R.T * (b / np.sum(R, axis=1))).T

        num_op = num_op + 2 * n * r + 2 * m * r + m + n + 2 * r

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

        num_op = num_op + 4 * n * r + 4 * m * r + 3 * n + 3 * m + 3 * r
    #####################################

    if gamma_init == "theory":
        L_trans = (
            (2 / (alpha) ** 4) * (np.linalg.norm(C1) ** 2) * (np.linalg.norm(C1) ** 2)
        )
        L_trans = (
            L_trans
            + ((reg + 2 * np.linalg.norm(C1) * np.linalg.norm(C1)) / (alpha**3)) ** 2
        )
        L = np.sqrt(3 * L_trans)
        gamma = 1 / L

    if gamma_init == "regularization":
        gamma = 1 / reg

    if gamma_init == "arbitrary":
        gamma = gamma_0

    # Classical OT
    C_trans = np.dot(C2, R)
    C_trans = np.dot(C1, C_trans)
    C_trans = C_trans / g
    G = np.dot(Q.T, C_trans)
    OT_trans = np.trace(G)

    acc.append(OT_trans)
    list_num_op.append(num_op)
    time_actual = time.time() - start
    times.append(time_actual)

    err = 1
    niter = 0
    count_escape = 1
    while (niter < max_iter) and (time_actual < time_out):
        Q_prev = Q
        R_prev = R
        g_prev = g
        if err > delta:
            niter = niter + 1

            K1_trans_0 = np.dot(C2, R)  # d * m * r
            K1_trans_0 = np.dot(C1, K1_trans_0)  # n * d * r
            grad_Q = K1_trans_0 / g
            if reg != 0.0:
                grad_Q = grad_Q + reg * np.log(Q)

            if gamma_init == "rescale":
                norm_1 = np.max(np.abs(grad_Q)) ** 2

            K2_trans_0 = np.dot(C1.T, Q)  # d * n * r
            K2_trans_0 = np.dot(C2.T, K2_trans_0)  # m * d * r
            grad_R = K2_trans_0 / g
            if reg != 0.0:
                grad_R = grad_R + reg * np.log(R)

            if gamma_init == "rescale":
                norm_2 = np.max(np.abs(grad_R)) ** 2

            omega = np.diag(np.dot(Q.T, K1_trans_0))  # r * n * r
            grad_g = -omega / (g**2)
            if reg != 0.0:
                grad_g = grad_g + reg * np.log(g)

            if gamma_init == "rescale":
                norm_3 = np.max(np.abs(grad_g)) ** 2

            if gamma_init == "rescale":
                gamma = gamma_0 / max(norm_1, norm_2, norm_3)

            C1_trans = grad_Q - (1 / gamma) * np.log(Q)  # 3 * n * r
            C2_trans = grad_R - (1 / gamma) * np.log(R)  # 3 * m * r
            C3_trans = grad_g - (1 / gamma) * np.log(g)  # 4 * r

            num_op = (
                num_op
                + 2 * n * d * r
                + 2 * m * d * r
                + r * n * r
                + 3 * n * r
                + 3 * m * r
                + 4 * r
            )

            # Update the coupling
            if method == "IBP":
                K1 = np.exp((-gamma) * C1_trans)
                K2 = np.exp((-gamma) * C2_trans)
                K3 = np.exp((-gamma) * C3_trans)
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
                K3 = np.exp((-gamma) * C3_trans)
                num_op = num_op + 2 * n * r + 2 * m * r + 2 * r
                Q, R, g, count_op_Dysktra, n_iter_Dykstra = LR_Dykstra_Sin(
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

                num_op = num_op + count_op_Dysktra

            if method == "Dykstra_LSE":
                Q, R, g, count_op_Dysktra_LSE = LR_Dykstra_LSE_Sin(
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

                num_op = num_op + count_op_Dysktra_LSE

            # Classical OT cost
            C_trans = np.dot(C2, R)
            C_trans = np.dot(C1, C_trans)
            C_trans = C_trans / g
            G = np.dot(Q.T, C_trans)
            OT_trans = np.trace(G)

            if np.isnan(OT_trans) == True:
                print("Error LOT: OT cost", niter)
                Q = Q_prev
                R = R_prev
                g = g_prev
                break

            err_1 = ((1 / gamma) ** 2) * (KL(Q, Q_prev) + KL(Q_prev, Q))
            err_2 = ((1 / gamma) ** 2) * (KL(R, R_prev) + KL(R_prev, R))
            err_3 = ((1 / gamma) ** 2) * (KL(g, g_prev) + KL(g_prev, g))
            criterion = err_1 + err_2 + err_3
            # print(criterion)

            if niter > 1:
                if criterion > delta / 1e-1:
                    err = criterion
                else:
                    count_escape = count_escape + 1
                    if count_escape != niter:
                        err = criterion

            # ## Update the error: Practical error
            # err = np.abs(OT_trans - acc[-1]) / acc[-1]

            if np.isnan(criterion) == True:
                print("Error LOT: stopping criterion", niter)
                Q = Q_prev
                R = R_prev
                g = g_prev
                break

            acc.append(OT_trans)
            list_num_op.append(num_op)
            time_actual = time.time() - start
            times.append(time_actual)
            list_criterion.append(criterion)

        else:
            break

    return (
        acc[-1],
        np.array(acc),
        np.array(times),
        np.array(list_num_op),
        np.array(list_criterion),
        Q,
        R,
        g,
    )


def clustering_lin_LOT(
    X, cost, cost_factorized, num_cluster=2, gamma_0=1, C_init=False, time_out=100
):
    a = np.ones(np.shape(X)[0]) / np.shape(X)[0]
    results = Lin_LOT_MD(
        X,
        X,
        a,
        a,
        num_cluster,
        cost,
        cost_factorized,
        gamma_0=gamma_0,
        C_init=C_init,
        time_out=time_out,
        reg=0,
        alpha=1e-10,
        max_iter=1000,
        delta=1e-5,
        Init="general_kmeans",
        seed_init=49,
        reg_init=1e-1,
        gamma_init="rescale",
        method="Dykstra",
        max_iter_IBP=10000,
        delta_IBP=1e-3,
        lam_IBP=0,
    )
    res_q, acc_q, times_q, list_num_op_q, list_criterion, Q, R, g = results
    y_pred = np.argmax(Q, axis=1)
    return y_pred


def clustering_quad_LOT(X, cost, num_cluster=2, gamma_0=1, C_init=False, time_out=100):
    a = np.ones(np.shape(X)[0]) / np.shape(X)[0]
    results = Quad_LOT_MD(
        X,
        X,
        a,
        a,
        num_cluster,
        cost,
        gamma_0=gamma_0,
        C_init=C_init,
        time_out=time_out,
        reg=0,
        alpha=1e-10,
        max_iter=1000,
        delta=1e-5,
        Init="general_kmeans",
        seed_init=49,
        reg_init=1e-1,
        gamma_init="rescale",
        method="Dykstra",
        max_iter_IBP=10000,
        delta_IBP=1e-3,
        lam_IBP=0,
    )
    res_q, acc_q, times_q, list_num_op_q, Q, R, g = results
    y_pred = np.argmax(Q, axis=1)
    return y_pred
