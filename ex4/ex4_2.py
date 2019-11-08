import matplotlib.pyplot as plt
import random
import numpy as np


#数据处理


fp = open("training_data.txt", "r")
fq = open("test_data.txt", "r")
v = [3, 5, 4, 4, 3, 2, 3, 3]
data_x = []
for i in fp.readlines():
    now = []
    for j in range(9):
        now.append(int(i.split()[j]))
    data_x.append(now)
tot = len(data_x)

data_y = []
for i in fq.readlines():
    now = []
    for j in range(9):
        now.append(int(i.split()[j]))
    data_y.append(now)
tot_y = len(data_y)


def solve():
    cal = [0] * tot_y
    y = [0] * 10
    for i in range(tmp):
        y[data[i][8]] = y[data[i][8]] + 1

    x = [[[0 for i in range(10)] for j in range(10)] for k in range(10)]

    for i in range(tmp):
        for j in range(8):
            x[j][data[i][j]][data[i][8]] = x[j][data[i][j]][data[i][8]] + 1

    for i in range(8):
        for j in range(10):
            for k in range(5):
                x[i][j][k] = (x[i][j][k] + 1) / (y[k] + v[i])
    for i in range(10):
        y[i] = (y[i] + 1) / (tot + 5)
    cnt = 0
    for i in range(tot_y):
        mx = 0
        for j in range(5):
            now = y[j]
            for k in range(8):
                now = now * x[k][data_y[i][k]][j]
            if now > mx:
                mx = now
                cal[i] = j
        if cal[i] != data_y[i][8]:
            cnt = cnt + 1

    return 1 - cnt / tot_y


x = []
y = []
for i in range(int(tot / 100)):
    data = []
    tmp = int((i + 1) * (tot / 100))
    mp = {}
    for j in range(tmp):
        pos = random.randint(0, tot - 1)
        while pos in mp:
            pos = random.randint(0, tot - 1)
        mp[pos] = 1
        data.append(data_x[pos])
    x.append(tmp)
    y.append(solve())

x = np.asarray(x)
y = np.asarray(y)

plt.plot(x, y, c='r')
plt.show()



