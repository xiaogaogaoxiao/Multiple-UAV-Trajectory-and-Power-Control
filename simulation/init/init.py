import csv

va = 5
vd = 5
vl = 20
hmin = 100
hmax = 170
time_slot = 160
K = 4

s = [[300, 0, 0], [100, 600, 0], [700, 700, 0], [100, 800, 0]]
ini = [[0, 0, 100], [30, 0, 100], [0, 30, 100], [30, 30, 100]]
dl = []
dh = []
desth = [100, 120, 140, 160]
data = [[], [], [], []]


def main():
    for k in range(K):
        dl.append([(s[k][0] - ini[k][0]) / time_slot, (s[k][1] - ini[k][1]) / time_slot])
    a_times = [0, 8, 16, 24]
    d_time = 135
    for i in range(time_slot):
        for k in range(K):
            if i < a_times[k]:
                # ascending
                data[k].append([ini[k][0] + dl[k][0] * i,
                                ini[k][1] + dl[k][1] * i, ini[k][2] + va * i])
            elif i < d_time or k == 0:
                data[k].append([ini[k][0] + dl[k][0] * i, ini[k][1] + dl[k][1] * i, desth[k]])
            else:
                data[k].append([ini[k][0] + dl[k][0] * i, ini[k][1] + dl[k][1] * i,
                                desth[k] - (i - d_time) * vd if desth[k] - (i - d_time) * vd > hmin else hmin])
    for k in range(K):
        with open('init_trajectory{}.csv'.format(k), mode='w') as csv_file:
            wr = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for d in data[k]:
                wr.writerow(d)


if __name__ == '__main__':
    main()
