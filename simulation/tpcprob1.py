import cvxpy as cp
import numpy as np
from scipy.linalg import eigh as largest_eigh
import math
from math import sqrt

itertime = 30
tdv = 300
M = 30
K = 4
e = 1
pmax = 30
dmin = 0
vl = 20
va = 5
vd = 5
hmin = 100
hmax = 170
ep = 0.001
b_0 = -50
n_0 = -160
bandwidth = 10
gamma = b_0 / (bandwidth * n_0)

s = np.array([[-250, -250, 0], [-250, 250, 0], [250, -250, 0], [250, 250, 0]])  # initial base station
ip = np.array([[0, 0, hmin], [30, 0, hmin], [0, 30, hmin], [30, 30, hmin]])


def cvx_hoovering():
    ini = np.array([[-250, -250, hmin], [-250, 250, hmin], [250, -250, hmin], [250, 250, hmin]])
    ipc = np.array([[-250, -250, hmin], [-250, 250, hmin], [250, -250, hmin], [250, 250, hmin]])
    ac = [math.sqrt(pmax) * 0.95 for k in range(K)]
    ir = [None] * K

    while True:
        print(ac)
        print(ipc)
        R_r = 0
        for k in range(K):
            addnoise = 1
            for j in range(K):
                if j != k:
                    addnoise += gamma * (ac[j] ** 2) / (np.linalg.norm(ipc[j] - s[k]) ** 2)
            R_r += bandwidth * math.log(1 + (gamma * (ac[k] ** 2)) / (np.linalg.norm(ipc[k] - s[k]) ** 2 * addnoise), 2)

        for k in range(K):
            ik = 0
            for j in range(K):
                if j != k:
                    ik += gamma * (ac[j] ** 2) / ((np.linalg.norm(ipc[j] - s[k])) ** 2)
            ir[k] = ik

        a = cp.Variable(shape=(K, 1))
        q = cp.Variable(shape=(K, 3))

        objfunc = []
        for k in range(K):
            term1 = 1
            for j in range(K):
                term1 += gamma * (2 * ac[j] * a[j] / (cp.norm(ipc[j] - s[k]) ** 2))
                term1 -= gamma * (ac[j] ** 2) * (cp.norm(q[j] - s[k]) ** 2) / (np.linalg.norm(ipc[j] - s[k]) ** 4)

            objfunc.append(cp.log(term1))
            objfunc.append(-1 * math.log(1 + ir[k], 2))
            objfunc.append(ir[k] / (1 + ir[k]))

            term2 = []
            for j in range(K):
                ratio = -1 * gamma / (1 + ir[k])
                if j != k:
                    det = np.linalg.norm(ipc[j] - s[k]) ** 2 + 2 * (ipc[j] - s[k]).transpose() * (q[j] - ipc[j])
                    term2.append(ratio * cp.quad_over_lin(a[j], det))
            objfunc.append(cp.sum(term2))

        constr = []
        for k in range(K):
            constr.append(q[k][2] >= hmin)
            constr.append(q[k][2] <= hmax)
            constr.append(cp.norm(q[k][0:2] - ini[k][0:2]) <= 0.5 * M * tdv * vl)
            constr.append(a[k] >= 0)
            constr.append(a[k] <= math.sqrt(pmax))
            for j in range(k + 1, K):
                constr.append(
                    2 * (ipc[k] - ipc[j]).transpose() * (q[k] - q[j]) >= cp.norm(ipc[j] - s[k]) ** 2 + dmin ** 2)

        prob = cp.Problem(cp.Maximize(sum(objfunc)), constr)
        prob.solve()
        ac = a.value
        ipc = q.value


def cvxprob(ar, qr, ir):
    a = []
    q = []
    for n in range(M):
        a.append(cp.Variable(shape=(K, 1)))
        q.append(cp.Variable(shape=(K, 3)))

    objfunc = []
    for n in range(M):
        for k in range(K):
            term1 = 1
            for j in range(K):
                term1 += gamma * (2 * ar[n][j] * a[n][j] / (np.linalg.norm(qr[n][j] - s[k])))
                term1 -= gamma * (ar[n][j] ** 2) * (cp.norm(q[n][j] - s[k]) ** 2) / (
                        np.linalg.norm(qr[n][j] - s[k]) ** 2)

            objfunc.append(cp.log(term1))
            objfunc.append(-1 * cp.log(1 + ir[n][k][0]))
            objfunc.append(ir[n][k][0] / (1 + ir[n][k][0]))

            term2 = 0
            for j in range(K):
                if j != k:
                    term2 -= gamma * cp.square(a[n][j]) / ((1 + ir[n][k][0]) * (
                            np.linalg.norm(qr[n][j] - s[k]) + 2 * (qr[n][j] - s[k]).transpose() * (
                            q[n][j] - qr[n][j])))
            objfunc.append(term2)

    constr = []
    for n in range(M):
        for k in range(K):
            constr.append(q[n][k][2] <= hmax)
            constr.append(q[n][k][2] >= hmin)

    for n in range(1, M):
        for k in range(K):
            constr.append(cp.norm(q[n][k][0:1] - q[n - 1][k][0:1]) <= vl)
            constr.append(q[n][k][2] - q[n][k - 1][2] <= va)
            constr.append(q[n][k][2] - q[n][k - 1][2] >= -vd)

    for n in range(M):
        for k in range(K):
            constr.append(a[n][k] <= math.sqrt(pmax))
            constr.append(a[n][k] >= 0)

    for n in range(M):
        for k in range(K):
            for j in range(k + 1, K):
                if j != k:
                    constr.append(2 * (qr[n][k] - qr[n][j]).transpose() * (q[n][k] - q[n][j]) >= (
                            np.linalg.norm(qr[n][j] - s[k]) ** 2 + dmin ** 2))

    obj = cp.Maximize(sum(objfunc))
    prob = cp.Problem(obj, constr)
    prob.solve()
    print("status: ", prob.status)
    return [av.value for av in a], [qv.value for qv in q]


def main():
    # Initialize q and a
    # q = []
    # for k in range(K):
    #     q.append(ip)
    # for i in range(1, 6):
    #     q.append(np.array([
    #         [500, 500 - 50 * i, hmin],
    #         [500 - 50 * i, -500, hmin],
    #         [-500 + 50 * i, 500, hmin],
    #         [-500, -500 + 50 * i, hmin]
    #     ]))
    # deriv = [((s[k][0] - q[5][k][0]) / 14, (s[k][1] - q[5][k][1]) / 14) for k in range(K)]
    # for i in range(1, 25):
    #     q.append(np.array([
    #         [q[5][k][0] + i * deriv[k][0], q[5][k][1] + i * deriv[k][1], hmin]
    #         for k in range(K)
    #     ]))
    # a = [np.array([[math.sqrt(pmax)] for k in range(K)])] * M
    # R = []
    # for n in range(M):
    #     for k in range(K):
    #         addnoise = 1
    #         for j in range(K):
    #             if j != k:
    #                 addnoise += gamma * (a[n][j] ** 2) / (np.linalg.norm(q[0][j] - s[k]) ** 2)
    #         R.append(
    #             bandwidth * math.log((1 + gamma * (a[n][k] ** 2) / ((np.linalg.norm(q[n][k] - s[k]) ** 2) * addnoise))))
    #
    # trace_r = [sum(R)]
    # count_iter = 0
    # while True:
    #     print(count_iter)
    #     count_iter += 1
    #     ir = [[None] * K] * M
    #
    #     # update ir
    #     for n in range(M):
    #         for k in range(K):
    #             ik = 0
    #             for j in range(K):
    #                 if j != k:
    #                     ik += gamma * a[n][j] * a[n][j] / (np.linalg.norm(q[n][j] - s[k]))
    #             ir[n][k] = ik
    #
    #     a, q = cvxprob(a, q, ir)
    #
    #     r_r = []
    #     for n in range(M):
    #         for k in range(K):
    #             addnoise = 1
    #             for j in range(K):
    #                 if j != k:
    #                     addnoise += gamma * (a[n][j] ** 2) / (np.linalg.norm(q[0][j] - s[k]) ** 2)
    #             r_r.append(bandwidth * math.log(
    #                 (1 + gamma * (a[n][k] ** 2) / ((np.linalg.norm(q[n][k] - s[k]) ** 2) * addnoise))))
    #     # break
    #     print(sum(r_r))
    #     if (sum(r_r) - trace_r[-1]) / trace_r[-1] < ep:
    #         break
    #
    #     trace_r.append(sum(r_r))
    cvx_hoovering()


if __name__ == '__main__':
    main()
