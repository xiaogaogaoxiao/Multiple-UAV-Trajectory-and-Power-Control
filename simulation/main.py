import numpy as np
import cvxpy as cp
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

UAV_num = 5
dim = 3
sample_num = 30  # count of time slots
bandwidth = 25000000  # communication bandwidth of FDMA
psd = 1  # power spectral density of AWGN
b = -50 * math.log(10, 2)  # channel power gain coefficient
y = b / (bandwidth * psd)

e = 0.1  # iteration tolerance
s = [0] * UAV_num  # location of base stations


def channel_gain(p_i, q_i, s_k):
    """
    Channel gain between ith UAV and kth base station.
    The calculation is modeled by the free-space path loss
    model.

    :param p_i: location of ith UAV
    :type p_i: [dim, 1] vector

    :param q_i: transmission power of ith UAV
    :type q_i: scalar

    :param s_k: location of kth base station
    :type s_k: [dim, 1] vector

    :return: channel gain between ith UAV and kth base station
    """
    return y * p_i / (np.linalg.norm(q_i - s_k) ** 2)


def achievable_rate(p, q, k):
    """
    Achievable rate (bps) of the UAV at a fixed time slot

    :param p: power vector at time slot N
    :type p:

    :param q: location vector
    :type q: location vector at time slot N

    :param k: index of the UAV
    :type k: scalar

    :return: R
    """
    addictive_noise = 1
    for i in range(len(p)):
        if i != k:
            addictive_noise += channel_gain(p[i], q[i], s[k])

    var = 1 + channel_gain(p[k], q[k], s[k]) / addictive_noise
    return bandwidth * math.log(2) * var


def cvx_problem_23(a_r, q_r, p_r, d, I):
    """
    Convex optimization for problem 23

    :param a_r: _a_ list in the last iteration
    :type a_r: [1, UAV_num] * N (list)

    :param q_r: trajectory in the last iteration
    :type q_r: [dim, UAV_num] * N (list)

    :param p_r: power vectors in the last iteration
    :type p_r: [1, UAV_num] * N (list)

    :param I: I in the last iteration
    :type I: [1, UAV_num] * N (list)

    :param d: distance matrices in the last iteration
    :type d: [UAV_num, UAV_num] * N (list)

    :return: a^{r+1}, q^{r+1}
    """
    a = [cp.Variable(1, UAV_num)] * sample_num
    q = [cp.Variable((dim, UAV_num))] * sample_num
    obj_func = 0
    for n in range(sample_num):
        for k in range(UAV_num):
            term1 = 1
            for j in range(UAV_num):
                term1 += y * (2 * a_r[n][j]) / d[n][j][k] - p_r[n][j] * (cp.norm(q[n][j] - s[k]) ** 2) / (
                        d[n][j][k] ** 2)

            obj_func += cp.log(term1)
            obj_func -= cp.log(I[n][k] + 1) + I[n][k] / (1 + I[n][k])

            term2 = 0
            for j in range(UAV_num):
                if j != k:
                    j += cp.square(a[n][j]) / (
                            d[n][j][k] + 2 * np.transpose((q_r[n][j] - s[k])) * (q[n][j] - q_r[n][j]))
            obj_func -= term2 * y / (1 + I[n][k])

    # TODO: add constraints
    constr = []

    prob = cp.Problem(cp.Maximize(obj_func))
    prob.solve()

    return a, q


def tpc_algorithm(q_0, p_0):
    """
    SCA-based TPC algorithm

    :param q_0: initial trajectory vector (q_0[N][k] -> 3 dim vector)
    :type q_0: N size list with at each time slot the trajectory of each
    UAV is represented by a [dim, UAV_num] matrix

    :param p_0: initial power vector (p_0[N][k] -> scalar)
    :type p_0: N size list with each time slot the power of each UAV
    is represented by a [1, UAV_num] vector

    :return: q_r, p_r
    """

    # Calculate initial R^0
    R_0 = 0
    for n in range(sample_num):
        for k in range(UAV_num):
            R_0 += achievable_rate(q_0[n][k], p_0[n][k], k)

    # Init
    p = p_0
    q = q_0
    R = [R_0]  # trace of R

    while True:
        # Update 1
        a = [[]] * sample_num
        I = [[]] * sample_num
        d = []
        for n in range(sample_num):
            for k in range(UAV_num):
                a[n][k] = math.sqrt(p[n][k])

        for n in range(sample_num):
            for k in range(UAV_num):
                for j in range(UAV_num):
                    if k != j:
                        I[n][k] += y * (a[n][j] ** 2) / (np.linalg.norm(q[n][j] - s[k]) ** 2)

        # Update 2
        a, q = cvx_problem_23(a, q, p, d, I)
        for n in range(sample_num):
            for k in range(UAV_num):
                p[n][k] = a[n][k] ** 2

        # Update 3
        R_r = 0
        for n in range(sample_num):
            for k in range(UAV_num):
                R_r += achievable_rate(q[n][k], p[n][k], k)
        R.append(R_r)

        if (R[-1] - R[-2]) / R[-2] < e:
            break


def main():
    # TODO: initialization of p, q vector
    # TODO(2): initialization of base stations
    pass


if __name__ == '__main__':
    main()
