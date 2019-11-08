import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#数据处理
fp = open("training_data.txt", "r")
fq = open("test_data.txt", "r")

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
cal = [0] * tot_y
y = [0] * 10
for i in range(tot):
    y[data_x[i][8]] = y[data_x[i][8]] + 1
x = [[[0 for i in range(10)] for j in range(10)] for k in range(10)]

for i in range(8):
    for j in range(tot):
        x[i][data_x[j][i]][data_x[j][8]] = x[i][data_x[j][i]][data_x[j][8]] + 1


for i in range(8):
    for j in range(5):
        for k in range(5):
            x[i][j][k] = x[i][j][k] / y[k]

for i in range(5):
    y[i] = y[i] / tot

cnt = 0
for i in range(tot_y):
    mx = 0
    for j in range(5):
        now = y[j]
        for k in range(8):
            now = now * x[k][data_y[i][k]][j]
        # if i < 10 and j == 0:
        #     print(i,j,now)
        if now > mx:
            mx = now
            cal[i] = j
    if cal[i] != data_y[i][8]:
        cnt = cnt + 1
print(cnt, tot_y)
print(1 - cnt/tot_y)







