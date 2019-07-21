import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

it = 0
sp_num = 160
b_0 = -50
n_0 = -160
bandwidth = 10000000
gamma = b_0 / (bandwidth * n_0)

s = [[300, 0, 0], [100, 600, 0], [700, 700, 0], [100, 800, 0]]
q = [[], [], [], []]

gs = gridspec.GridSpec(2, 2)
iter_time = 0
fig = plt.figure(tight_layout=True, figsize=(15, 12))
ax = fig.add_subplot(gs[0, 1], projection='3d')
ax_2 = fig.add_subplot(gs[0, 0], projection='3d')
x1 = []
y1 = []
z1 = []
with open('init_trajectory0.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        x1.append(float(row[0]))
        y1.append(float(row[1]))
        z1.append(float(row[2]))

tck, u = interpolate.splprep([x1, y1, z1], s=2)
u_fine = np.linspace(0, 1, sp_num)
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
for i in range(iter_time):
    tck, u = interpolate.splprep([x_fine, y_fine, z_fine], s=2)
    u_fine = np.linspace(0, 1, sp_num)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
for i in range(len(x_fine)):
    q[0].append([x_fine[i], y_fine[i], z_fine[i]])
#ax.plot(x_fine, y_fine, z_fine, c='blue', label='trajectory of UAV 1')
ax_2.plot(x_fine, y_fine, z_fine, c='blue', label='trajectory of UAV 1')
#ax.scatter(x1[0], y1[0], z1[0], marker='o', s=15, c='blue')
ax_2.scatter(x1[0], y1[0], z1[0], marker='o', s=15, c='blue')
xd, yd, zd = [x_fine[-1], s[0][0]], [y_fine[-1], s[0][1]], [z_fine[-1], s[0][2]]
#ax.plot(xd, yd, zd, linestyle=':', c='gray')
ax_2.plot(xd, yd, zd, linestyle=':', c='gray')

x1 = []
y1 = []
z1 = []
with open('init_trajectory1.csv'.format(it)) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        x1.append(float(row[0]))
        y1.append(float(row[1]))
        z1.append(float(row[2]))
tck, u = interpolate.splprep([x1, y1, z1], s=2)
u_fine = np.linspace(0, 1, sp_num)
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
for i in range(iter_time):
    tck, u = interpolate.splprep([x_fine, y_fine, z_fine], s=2)
    u_fine = np.linspace(0, 1, sp_num)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
#ax.plot(x_fine, y_fine, z_fine, c='orange', label='trajectory of UAV 2')
ax_2.plot(x_fine, y_fine, z_fine, c='orange', label='trajectory of UAV 2')
#ax.scatter(x1[0], y1[0], z1[0], marker='o', s=15, c='orange')
ax_2.scatter(x1[0], y1[0], z1[0], marker='o', s=15, c='orange')
xd, yd, zd = [x_fine[-1], s[1][0]], [y_fine[-1], s[1][1]], [z_fine[-1], s[1][2]]
#ax.plot(xd, yd, zd, linestyle=':', c='gray')
ax_2.plot(xd, yd, zd, linestyle=':', c='gray')
for i in range(len(x_fine)):
    q[1].append([x_fine[i], y_fine[i], z_fine[i]])

x1 = []
y1 = []
z1 = []
with open('init_trajectory2.csv'.format(it)) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        x1.append(float(row[0]))
        y1.append(float(row[1]))
        z1.append(float(row[2]))
tck, u = interpolate.splprep([x1, y1, z1], s=2)
u_fine = np.linspace(0, 1, sp_num)
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
for i in range(iter_time):
    tck, u = interpolate.splprep([x_fine, y_fine, z_fine], s=2)
    u_fine = np.linspace(0, 1, sp_num)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
#ax.plot(x_fine, y_fine, z_fine, c='green', label='trajectory of UAV 3')
ax_2.plot(x_fine, y_fine, z_fine, c='green', label='trajectory of UAV 3')
#ax.scatter(x_fine[0], y_fine[0], z_fine[0], marker='o', s=15, c='green')
ax_2.scatter(x_fine[0], y_fine[0], z_fine[0], marker='o', s=15, c='green')
xd, yd, zd = [x_fine[-1], s[2][0]], [y_fine[-1], s[2][1]], [z_fine[-1], s[2][2]]
#ax.plot(xd, yd, zd, linestyle=':', c='gray')
ax_2.plot(xd, yd, zd, linestyle=':', c='gray')
for i in range(len(x_fine)):
    q[2].append([x_fine[i], y_fine[i], z_fine[i]])

x1 = []
y1 = []
z1 = []
with open('init_trajectory3.csv'.format(it)) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        x1.append(float(row[0]))
        y1.append(float(row[1]))
        z1.append(float(row[2]))
tck, u = interpolate.splprep([x1, y1, z1], s=2)
u_fine = np.linspace(0, 1, sp_num)
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
for i in range(iter_time):
    tck, u = interpolate.splprep([x_fine, y_fine, z_fine], s=2)
    u_fine = np.linspace(0, 1, sp_num)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
#ax.plot(x_fine, y_fine, z_fine, c='red', label='trajectory of UAV 4')
ax_2.plot(x_fine, y_fine, z_fine, c='red', label='trajectory of UAV 4')
for i in range(len(x_fine)):
    q[3].append([x_fine[i], y_fine[i], z_fine[i]])
#ax.scatter(x1[0], y1[0], z1[0], marker='o', s=15, c='red')
ax_2.scatter(x1[0], y1[0], z1[0], marker='o', s=15, c='red')
xd, yd, zd = [x_fine[-1], s[3][0]], [y_fine[-1], s[3][1]], [z_fine[-1], s[3][2]]
#ax.plot(xd, yd, zd, linestyle=':', c='gray')
ax_2.plot(xd, yd, zd, linestyle=':', c='gray')

#ax.scatter(s[0][0], s[0][1], s[0][2], marker='^', s=100, c='blue', label='BS 1')
#ax.scatter(s[1][0], s[1][1], s[1][2], marker='^', s=100, c='orange', label='BS 2')
#ax.scatter(s[2][0], s[2][1], s[2][2], marker='^', s=100, c='green', label='BS 3')
#ax.scatter(s[3][0], s[3][1], s[3][2], marker='^', s=100, c='red', label='BS 4')
ax_2.scatter(s[0][0], s[0][1], s[0][2], marker='^', s=100, c='blue', label='BS 1')
ax_2.scatter(s[1][0], s[1][1], s[1][2], marker='^', s=100, c='orange', label='BS 2')
ax_2.scatter(s[2][0], s[2][1], s[2][2], marker='^', s=100, c='green', label='BS 3')
ax_2.scatter(s[3][0], s[3][1], s[3][2], marker='^', s=100, c='red', label='BS 4')

# #ax.legend(ncol=2)
#ax.set_xlabel('$X$')
#ax.set_ylabel('$Y$')
#ax.set_zlabel('$Z$')
#ax.view_init(elev=0., azim=140)

ax_2.legend(ncol=2)
ax_2.set_xlabel('$X$')
ax_2.set_ylabel('$Y$')
ax_2.set_zlabel('$Z$')
ax_2.view_init(elev=50., azim=120)

bx = fig.add_subplot(gs[1, :])
achievable_rate = [[], [], [], []]
for n in range(sp_num):
    for k in range(4):
        addictive_noise = 1
        for j in range(4):
            if j != k:
                addictive_noise += gamma * 10 / (np.linalg.norm(np.array(q[j][n]) - np.array(s[k])) ** 2)
        r = bandwidth * math.log(
            1 + gamma * 10 / ((np.linalg.norm(np.array(q[k][n]) - np.array(s[k])) ** 2) * addictive_noise), 2)
        achievable_rate[k].append(r * 2048)

xs = np.linspace(0, sp_num, sp_num)
bx.scatter(xs, achievable_rate[0], c='blue', alpha='0.3', s=5)
bx.plot(xs, achievable_rate[0], c='blue', alpha=0.4, linestyle="-.", label='UAV 1 achievable rate')
bx.scatter(xs, achievable_rate[1], c='orange', alpha='0.3', s=5)
bx.plot(xs, achievable_rate[1], c='orange', alpha=0.4, linestyle="-.", label='UAV 2 achievable rate')
bx.scatter(xs, achievable_rate[2], c='green', alpha='0.3', s=5)
bx.plot(xs, achievable_rate[2], c='green', alpha=0.4, linestyle="-.", label='UAV 3 achievable rate')
bx.scatter(xs, achievable_rate[3], c='red', alpha='0.3', s=5)
bx.plot(xs, achievable_rate[3], c='red', alpha=0.4, linestyle="-.", label='UAV 4 achievable rate')
bx.plot(xs, [sum([ar[n] for ar in achievable_rate]) for n in range(160)], c='black', linewidth='2.0',
        label='Aggregate achievable rate')
bx.set_ylabel("Transmission Rate (bps/Hz)")
bx.set_xlabel("Time Slot")
bx.legend()
bx.set_title('Initial Achievable Transmission Rate', y=-0.15)

cx = fig.add_subplot(gs[0, 1])
xh = np.linspace(0, 160, 160)

cx.plot(xh, [float(d[2]) if float(d[2]) > 100 else 200 - (float(d[2])) for d in q[0]][:160], c='blue',
        label='altitude of UAV 1')
cx.plot(xh, [float(d[2]) for d in q[1]][:160], c='orange', label='altitude of UAV 2')
cx.plot(xh, [float(d[2]) for d in q[2]][:160], c='green', label='altitude of UAV 3')
cx.plot(xh, [float(d[2]) for d in q[3]][:160], c='red', label='altitude of UAV 4')
cx.set_ylabel("Altitude (meter)")
cx.set_xlabel("Time Slot")
cx.legend(loc=1)
print([sum([ar[n] for ar in achievable_rate]) for n in range(sp_num)])
plt.show()
