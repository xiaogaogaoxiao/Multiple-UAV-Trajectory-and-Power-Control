import cvxpy as cp
import numpy as np
from scipy.linalg import eigh as largest_eigh
import math

itertime = 30
M = 20
K = 4
e = 1
pmax = 30
gamma = 0.03125
ts = 0.5
dmin = 20
vl = 600
va = 150
hmin = 100
hmax = 101

s = np.array([[-250, -250, 0], [-250, 250, 0], [250, -250, 0], [250, 250, 0]])  # initial base station
ip = np.array([[500, 500, hmin], [500, -500, hmin], [-500, 500, hmin], [-500, -500, hmin]])


def cvxprob(ar, qr, pr, ir, q_q, c, k):
    a = [cp.Variable(shape=(K, 1))] * M
    q = [cp.Variable(shape=(K, 3))] * M
    objfunc = []
    for n in range(M):
        # term 1
        for j in range(K):
            ukj = (e + pr[n][j] / np.linalg.norm(qr[n][j] - s[k]))
            dt = K * e
            for i in range(K):
                dt += (pr[n][i] / np.linalg.norm(qr[n][i] - s[k]))
            ukj = ukj / dt
            objfunc.append(
                ukj * (cp.log(1 + gamma * (
                        2 * ar[n][k] * a[n][k] / np.linalg.norm(qr[n][k], s[k]) - pr[n][k] * (
                    cp.norm(q[n][k] - s[j] ** 2)) / ukj)))
            )

        # term 2 and 3
        objfunc.append(-1 * math.log(ir[n][k]))
        objfunc.append(ir[n][k] / (1 + ir[n][k]))

        # term 4
        for j in range(K):
            if j != k:
                objfunc.append(gamma * (a[n][k] ** 2) / (
                        (1 + ir[n][j]) * (np.linalg.norm(qr[n][k] - s[k]) + 2 * (np.transpose(qr[n][k] - s[j])) * (
                        q[n][k] - qr[n][k]))))

    for n in range(M):
        objfunc.append(c[k] * (cp.norm(q[n][k] - q_q[n][k]) ** 2) / -2)

    constr = []
    for n in range(M):
        for k in range(K):
            constr.append(q[n][k][2] <= hmax)
            constr.append(q[n][k][2] >= hmin)

    for n in range(1, M):
        for k in range(K):
            constr.append(cp.norm(q[n][k][0:1] - q[n - 1][k][0:1]) <= vl)
            constr.append(q[n][k][2] - q[n][k - 1][2] <= va)

    for n in range(M):
        for k in range(K):
            constr.append(a[n][k] <= math.sqrt(pmax))

    for k in range(K):
        constr.append(q[M - 1][k] == s[k])
        constr.append(q[0][k] == ip[k])

    prob = cp.Problem(cp.Maximize(cp.sum(objfunc)), constraints=constr)
    result = prob.solve()


def main():
    lamb = [np.full(shape=(K, K), fill_value=1)] * M
    b = np.full(shape=(K, K), fill_value=0.001)
    Al = []
    for k in range(K):
        Ai = np.dot(-1, np.identity(K - k - 1))
        A0 = np.full(shape=(K - k - 1, k), fill_value=0)
        A1 = np.full(shape=(K - k - 1, 1), fill_value=1)
        Aa = np.concatenate((A0, A1, Ai), axis=1)
        Al.append(Aa)
    A = np.concatenate(Al, axis=0)
    A = np.asmatrix(A)
    A = np.kron(A, np.identity(3))
    c = 1.1 * b * max(np.linalg.eigvals(np.dot(A.transpose(), A)))

    q = []
    for k in range(K):
        q.append(ip)
    for i in range(1, 6):
        q.append(np.array([
            [500, 500 - 50 * i, hmin],
            [500 - 50 * i, -500, hmin],
            [-500 + 50 * i, 500, hmin],
            [-500, -500 + 50 * i, hmin]
        ]))
    deriv = [((s[k][0] - q[5][k][0]) / 14, (s[k][1] - q[5][k][1]) / 14) for k in range(K)]
    for i in range(1, 15):
        q.append(np.array([
            [q[5][k][0] + i * deriv[k][0], q[5][k][1] + i * deriv[k][1], hmin]
            for k in range(K)
        ]))
    a = np.array([[pmax] for k in range(K)])

    # update z
    z = [np.ones(shape=(K, K, 3))] * M
    for n in range(M):
        for k in range(K):
            for j in range(K):
                mint = min(np.linalg.norm(q[n][k] - q[n][j] + (b[k][j] ** -1) * lamb[n][k][j]) / dmin, 1)
                z[n][k][j] = (q[n][k] - q[n][j] + (b[k][j] ** -1) * lamb[n][k][j]) / mint

    for iter in range(itertime):
        for k in range(K):
            q_q = [None] * M
            for n in range(M):
                print(A)
                q_q[n] = q[n][k] - (c[k] ** (-1)) * A[k].transpose() * b * (
                        A * q[n] - z[n] + np.linalg.inv(b) * lamb[n])


if __name__ == '__main__':
    main()
