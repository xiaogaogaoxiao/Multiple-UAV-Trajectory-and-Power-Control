import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

it = 11
sp_num = 120
b_0 = -50
n_0 = -160
bandwidth = 10000000
gamma = b_0 / (bandwidth * n_0)

s = [[300, 0, 0], [100, 600, 0], [700, 700, 0], [100, 800, 0]]
q = [[], [], [], []]

gs = gridspec.GridSpec(2, 2)
iter_time = 60
fig = plt.figure(tight_layout=True, figsize=(15, 12))
ax = fig.add_subplot(gs[0, 0], projection='3d')
x1 = []
y1 = []
z1 = []
with open('trajectory0_iteration{}.csv'.format(it)) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        x1.append(float(row[0]))
        y1.append(float(row[1]))
        z1.append(float(row[2]))
    csv_file.close()

tck, u = interpolate.splprep([x1, y1, z1], s=2)
u_fine = np.linspace(0, 1, sp_num)
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
for i in range(iter_time):
    tck, u = interpolate.splprep([x_fine, y_fine, z_fine], s=2)
    u_fine = np.linspace(0, 1, sp_num)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
for i in range(len(x_fine)):
    q[0].append([x_fine[i], y_fine[i], z_fine[i]])
ax.plot(x_fine, y_fine, z_fine, c='blue', label='trajectory of UAV 1')
ax.scatter(x_fine[0], y_fine[0], z_fine[0], marker='o', s=15, c='blue')
ax.scatter(x_fine[-1], y_fine[-1], z_fine[-1], marker='o', s=15, c='blue')
ax.text(x_fine[-1] + 170, y_fine[-1] + 50, z_fine[-1] - 10,
        '({}, {}, {})'.format(int(x_fine[-1]), int(y_fine[-1]), int(z_fine[-1])),
        fontsize=8)
xd, yd, zd = [x_fine[-1], s[0][0]], [y_fine[-1], s[0][1]], [z_fine[-1], s[0][2]]
ax.plot(xd, yd, zd, linestyle=':', c='gray')

x1 = []
y1 = []
z1 = []
with open('trajectory1_iteration{}.csv'.format(it)) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        x1.append(float(row[0]))
        y1.append(float(row[1]))
        z1.append(float(row[2]))
    csv_file.close()

tck, u = interpolate.splprep([x1, y1, z1], s=2)
u_fine = np.linspace(0, 1, sp_num)
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
for i in range(iter_time):
    tck, u = interpolate.splprep([x_fine, y_fine, z_fine], s=2)
    u_fine = np.linspace(0, 1, sp_num)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
ax.plot(x_fine, y_fine, z_fine, c='orange', label='trajectory of UAV 2')
ax.scatter(x_fine[0], y_fine[0], z_fine[0], marker='o', s=15, c='orange')
ax.scatter(x_fine[-1], y_fine[-1], z_fine[-1], marker='o', s=15, c='orange')
ax.text(x_fine[-1] - 15, y_fine[-1], z_fine[-1] - 10,
        '({}, {}, {})'.format(int(x_fine[-1]), int(y_fine[-1]), int(z_fine[-1])),
        fontsize=8)
xd, yd, zd = [x_fine[-1], s[1][0]], [y_fine[-1], s[1][1]], [z_fine[-1], s[1][2]]
ax.plot(xd, yd, zd, linestyle=':', c='gray')
for i in range(len(x_fine)):
    q[1].append([x_fine[i], y_fine[i], z_fine[i]])

x1 = []
y1 = []
z1 = []
with open('trajectory2_iteration{}.csv'.format(it)) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        x1.append(float(row[0]))
        y1.append(float(row[1]))
        z1.append(float(row[2]))
    csv_file.close()

tck, u = interpolate.splprep([x1, y1, z1], s=2)
u_fine = np.linspace(0, 1, sp_num)
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
for i in range(iter_time):
    tck, u = interpolate.splprep([x_fine, y_fine, z_fine], s=2)
    u_fine = np.linspace(0, 1, sp_num)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
ax.plot(x_fine, y_fine, z_fine, c='green', label='trajectory of UAV 3')
ax.scatter(x_fine[0], y_fine[0], z_fine[0], marker='o', s=15, c='green')
ax.scatter(x_fine[-1], y_fine[-1], z_fine[-1], marker='o', s=15, c='green')
ax.text(x_fine[-1] - 15, y_fine[-1], z_fine[-1] - 10,
        '({}, {}, {})'.format(int(x_fine[-1]), int(y_fine[-1]), int(z_fine[-1])),
        fontsize=8)
xd, yd, zd = [x_fine[-1], s[2][0]], [y_fine[-1], s[2][1]], [z_fine[-1], s[2][2]]
ax.plot(xd, yd, zd, linestyle=':', c='gray')
for i in range(len(x_fine)):
    q[2].append([x_fine[i], y_fine[i], z_fine[i]])

x1 = []
y1 = []
z1 = []
with open('trajectory3_iteration{}.csv'.format(it)) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        x1.append(float(row[0]))
        y1.append(float(row[1]))
        z1.append(float(row[2]))
    csv_file.close()

tck, u = interpolate.splprep([x1, y1, z1], s=2)
u_fine = np.linspace(0, 1, sp_num)
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
for i in range(iter_time):
    tck, u = interpolate.splprep([x_fine, y_fine, z_fine], s=2)
    u_fine = np.linspace(0, 1, sp_num)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
ax.plot(x_fine, y_fine, z_fine, c='red', label='trajectory of UAV 4')
for i in range(len(x_fine)):
    q[3].append([x_fine[i], y_fine[i], z_fine[i]])
ax.scatter(x_fine[0], y_fine[0], z_fine[0], marker='o', s=15, c='red')
ax.scatter(x_fine[-1], y_fine[-1], z_fine[-1], marker='o', s=15, c='red')
ax.text(x_fine[-1] + 150, y_fine[-1] + 50, z_fine[-1] - 10,
        '({}, {}, {})'.format(int(x_fine[-1]), int(y_fine[-1]), int(z_fine[-1])),
        fontsize=8)
xd, yd, zd = [x_fine[-1], s[3][0]], [y_fine[-1], s[3][1]], [z_fine[-1], s[3][2]]
ax.plot(xd, yd, zd, linestyle=':', c='gray')

ax.scatter(s[0][0], s[0][1], s[0][2], marker='^', s=100, c='blue', label='BS 1')
ax.scatter(s[1][0], s[1][1], s[1][2], marker='^', s=100, c='orange', label='BS 2')
ax.scatter(s[2][0], s[2][1], s[2][2], marker='^', s=100, c='green', label='BS 3')
ax.scatter(s[3][0], s[3][1], s[3][2], marker='^', s=100, c='red', label='BS 4')
ax.text(s[0][0] - 30, s[0][1], s[0][2], '({}, {}, {})'.format(int(s[0][0]), int(s[0][1]), int(s[0][2])), fontsize=8)
ax.text(s[1][0] - 15, s[1][1], s[1][2], '({}, {}, {})'.format(int(s[1][0]), int(s[1][1]), int(s[1][2])), fontsize=8)
ax.text(s[2][0] - 10, s[2][1] - 15, s[2][2] + 10, '({}, {}, {})'.format(int(s[2][0]), int(s[2][1]), int(s[2][2])),
        fontsize=8)
ax.text(s[3][0] + 150, s[3][1] + 50, s[3][2], '({}, {}, {})'.format(int(s[3][0]), int(s[3][1]), int(s[3][2])),
        fontsize=8)

ax.legend(ncol=2)
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')
ax.view_init(elev=50., azim=120)

bx = fig.add_subplot(gs[1, :])
achievable_rate = [[], [], [], []]
for n in range(sp_num):
    for k in range(4):
        addictive_noise = 1
        for j in range(4):
            if j != k:
                addictive_noise += gamma * 30 / (np.linalg.norm(np.array(q[j][n]) - np.array(s[k])) ** 2)
        r = bandwidth * math.log(
            1 + gamma * 30 / ((np.linalg.norm(np.array(q[k][n]) - np.array(s[k])) ** 2) * addictive_noise), 2)
        achievable_rate[k].append(r * 2048)

init_data = [0.14187991361827484, 0.14329044637643112, 0.14470753975206713, 0.14614431498014246, 0.14758109020821772,
             0.14905066852369192, 0.1505596105440449, 0.1521866436790343, 0.1539055254587409, 0.15565064970836662,
             0.15744169828035093, 0.15928523179217352, 0.1611746896263547, 0.1630969505479348, 0.165045453939434,
             0.16702676041833212, 0.1690736730720282, 0.17119931313548178, 0.17339711999121318, 0.17565397240426267,
             0.17795018852219086, 0.1802792077275179, 0.18265415125520343, 0.18508157972272726, 0.18754837189512974,
             0.19006764900737047, 0.1926394110594495, 0.195257097433887, 0.1979207081306829, 0.20064992500227669,
             0.20342506619622885, 0.20625925294749908, 0.2091524852560873, 0.21210476312199353, 0.21510952592773797,
             0.21819301614323977, 0.2213289912985798, 0.22453057262871762, 0.22780432075113288, 0.23113055381338637,
             0.23453551428539718, 0.23802576278464513, 0.24157505684121097, 0.24520963892501388, 0.2489229484185741,
             0.25272154593937146, 0.25659887086992594, 0.2605614838277176, 0.2646028241952664, 0.2687491344424916,
             0.27298073271695367, 0.27731074025361246, 0.2817457176699475, 0.2862594224960397, 0.29090433967172735,
             0.2956411054921318, 0.3005025230446517, 0.3054557892418883, 0.3105468284062001, 0.31575595868514766,
             0.32107661946125116, 0.32652849258694994, 0.3321181386797238, 0.33784555773957253, 0.3437107497664964,
             0.34972027537797495, 0.355880695191488, 0.36218544858955576, 0.3686607780420973, 0.3752870016966732,
             0.38209036202320257, 0.38906429840420564, 0.39621537145716207, 0.4035567024170314, 0.4110948519012934,
             0.4188166986749885, 0.42675504582551543, 0.4348902115004349, 0.4432484381696658, 0.4518297258332083,
             0.4606471957260218, 0.4697074084655861, 0.47899724281694134, 0.4885626231024461, 0.49837730685218107,
             0.5084675365360654, 0.5188464333890583, 0.52951399741116, 0.5404833498373298, 0.5517610512850475,
             0.5633667836067522, 0.5753005468024437, 0.5875820227245614, 0.6002177719905849, 0.6132209158354736,
             0.6266045754941868, 0.6403818722016841, 0.6545724878104044, 0.6691895435553076, 0.6842396000538729,
             0.6997423391585392, 0.7157043214867868, 0.7321517895085341, 0.7491044250762202, 0.7665687888073248,
             0.7845776837892467, 0.8031311100219853, 0.82224874935798, 0.8419634048846292, 0.862288197836892,
             0.8832362494497281, 0.9048403628105357, 0.9271202197717541, 0.9500758203333827, 0.97373996758282,
             0.9981585858424242, 1.0233119932597559, 1.0492329929222126, 1.075967509152153, 1.103508981332097,
             1.1318836519319628, 1.1611177634216694, 1.1912244370361755, 1.222210233392961, 1.2540948343444638,
             1.2869176035955625, 1.3206457380588579, 1.355312040821748, 1.3908968300317932, 1.427334499514195,
             1.4644413519795205, 1.5025913426241149, 1.543424625817919, 1.5896835396674622, 1.6441169828967461,
             1.7062919547520934, 1.7747519981529885, 1.8482440351607945, 1.9268402325677814, 2.0105865146962985,
             2.0951725558621854, 2.180112870371951, 2.270819967646328, 2.3678974244934308, 2.468005886616068,
             2.5694527147044637, 2.67568879355293, 2.7878294281329943, 2.8975885585678145, 2.99732962611108,
             3.0930031108168676, 3.185901454328675, 3.2760574597339063, 3.3619949880996485, 3.4419557939413723,
             3.5140504194249598, 3.576336921776463, 3.6269455250142126, 3.6641639079760493, 3.686594659138833]

for i in range(sp_num, 160):
    for kag in achievable_rate:
        kag.append(kag[-1])

xs = np.linspace(0, 160, 160)
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
bx.plot(xs, init_data, c='black', linewidth='2.0', linestyle="-.",
        label='Initial aggregate achievable rate')
bx.set_ylabel("Transmission Rate (bps/Hz)")
bx.set_xlabel("Time Slot")
bx.legend()

cx = fig.add_subplot(gs[0, 1])
xh = np.linspace(0, 115, 115)

cx.plot(xh, [float(d[2]) if float(d[2]) > 100 else 200 - (float(d[2])) for d in q[0]][:115], c='blue',
        label='altitude of UAV 1')
cx.plot(xh, [float(d[2]) for d in q[1]][:115], c='orange', label='altitude of UAV 2')
cx.plot(xh, [float(d[2]) for d in q[2]][:115], c='green', label='altitude of UAV 3')
cx.plot(xh, [float(d[2]) for d in q[3]][:115], c='red', label='altitude of UAV 4')
cx.set_ylabel("Altitude (meter)")
cx.set_xlabel("Time Slot")
cx.legend()

ax.set_title('Optimized Trajectory', y=-0.08)
bx.set_title('Optimized Achievable Transmission Rate', y=-0.15)
plt.show()
