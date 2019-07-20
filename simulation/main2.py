import cvxpy as cp
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

M = 160
K = 4
e = 1
pmax = 30
dmin = 20
vl = 20
va = 5
vd = 5
hmin = 100
hmax = 200
ep = 0.0001
b_0 = -50
n_0 = -160
bandwidth = 10000000
gamma = b_0 / (bandwidth * n_0)
minimum_iter_time = 30

s = np.array([[300, 0, 0], [100, 600, 0], [700, 700, 0], [100, 800, 0]])
q_track = []


def read_initial_trajectory():
    ret = [[], [], [], []]
    for k in range(K):
        with open('init_trajectory{}.csv'.format(k)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                ret[k].append([float(data) for data in row])

    return [[ret[k][n] for k in range(K)] for n in range(M)]


def main():
    q = np.array(read_initial_trajectory())
    a = [[math.sqrt(pmax) * 0.95] * K] * M
    r_list = []
    iter_time = 0
    q_track.append(eval(str(np.ndarray.tolist(q))))
    while True:
        iter_time += 1

        # update rate evaluation
        r_l = []
        for n in range(M):
            r = 0
            for k in range(K):
                addictive_noise = 1
                for j in range(K):
                    if j != k:
                        addictive_noise += gamma * (a[n][j] ** 2) / (np.linalg.norm(q[n][j] - s[k]) ** 2)
                r += bandwidth * math.log(
                    1 + gamma * (a[n][k] ** 2) / ((np.linalg.norm(q[n][k] - s[k]) ** 2) * addictive_noise), 2)
            r_l.append(r * 1024)
        print(r_l)
        r_list.append(r_l)
        print(sum(r_l))

        if len(r_list) >= minimum_iter_time:
            if (sum(r_list[-1]) - sum(r_list[-2])) / sum(r_list[-2]) <= ep:
                break

        # update ir
        ir = [[None] * K] * M
        for n in range(M):
            for k in range(K):
                ik = 0
                for j in range(K):
                    if j != k:
                        ik += gamma * a[n][j] * a[n][j] / (np.linalg.norm(q[n][j] - s[k]) ** 2)
                ir[n][k] = ik

        # update power and trajectory
        av = []
        qv = []
        for n in range(M):
            av.append(cp.Variable(shape=(K, 1)))
            qv.append(cp.Variable(shape=(K, 3)))

        objfunc = []
        for n in range(M):
            for k in range(K):
                termk = 0

                term1 = 1
                for j in range(K):
                    term1 += gamma * (2 * a[n][j] * av[n][j]) / (np.linalg.norm(q[n][j] - s[k]) ** 2)
                    term1 -= gamma * (a[n][j] * a[n][j]) * (cp.norm(qv[n][j] - s[k]) ** 2) / (
                            np.linalg.norm(q[n][j] - s[k]) ** 4)

                termk += (cp.log(term1))
                termk += (-1 * math.log(1 + ir[n][k]))
                termk += (ir[n][k] / (1 + ir[n][k]))

                term2 = []
                for j in range(K):
                    ratio = -1 * gamma / (1 + ir[n][k])
                    if j != k:
                        det = cp.norm(q[n][j] - s[k]) ** 2 + 2 * (q[n][j] - s[k]).transpose() * (
                                qv[n][j] - q[n][j])
                        term2.append(ratio * cp.quad_over_lin(av[n][j], det))
                termk += cp.sum(term2)
                objfunc.append(termk)

        constr = []

        for k in range(K):
            constr.append(qv[M - 1][k][0] == read_initial_trajectory()[-1][k][0])
            constr.append(qv[M - 1][k][1] == read_initial_trajectory()[-1][k][1])
            constr.append(qv[M - 1][k][2] == read_initial_trajectory()[-1][k][2])
            constr.append(qv[0][k][0] == read_initial_trajectory()[0][k][0])
            constr.append(qv[0][k][1] == read_initial_trajectory()[0][k][1])
            constr.append(qv[0][k][2] == read_initial_trajectory()[0][k][2])

        for n in range(M):
            for k in range(K):
                constr.append(qv[n][k][2] <= hmax)
                constr.append(qv[n][k][2] >= hmin)
                constr.append(av[n][k] <= math.sqrt(pmax))
                constr.append(av[n][k] >= 0)

        for n in range(1, M):
            for k in range(K):
                constr.append(cp.norm(qv[n][k][0:2] - qv[n - 1][k][0:2]) <= 0.5 * vl)
                constr.append(qv[n][k][2] - qv[n - 1][k][2] <= 0.5 * va)
                constr.append(qv[n][k][2] - qv[n - 1][k][2] >= -0.5 * va)

        for n in range(M):
            for k in range(K):
                for j in range(k + 1, K):
                    constr.append(2 * (q[n][k] - q[n][j]).transpose() * (qv[n][k] - qv[n][j]) >= (dmin ** 2))

        obj = cp.Maximize(cp.sum(objfunc))
        prob = cp.Problem(obj, constr)
        prob.solve()
        print("Iteration: {}\tStatus: {}".format(iter_time, prob.status))
        a = [v.value for v in av]
        q = [v.value for v in qv]
        q_track.append([[qnk for qnk in qn] for qn in q])

    for i in range(len(q_track)):
        for k in range(K):
            with open('trajectory{}_iteration{}.csv'.format(k, i), mode='w') as csv_file:
                wr = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for n in range(M):
                    wr.writerow(q_track[i][n][k])


if __name__ == '__main__':
    main()
