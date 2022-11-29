from os.path import join

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


dirpath = '/home/roblab15/Documents/FMG_project/data'
Title1 = 'nopoint/nopoint_4'
Title2 = 'point/point_4'
fileName1 = join(dirpath, Title1) + '.csv'
fileName2 = join(dirpath, Title2) + '.csv'

df1 = pd.read_csv(fileName1)
df2 = pd.read_csv(fileName2)
print(df1)
print(df2)
x1 = df1['time']
y1 = df1.drop(['time'], axis=1, inplace=False)

x2 = df2['time']
y2 = df2.drop(['time'], axis=1, inplace=False)


plt.figure()
plt.cla()
plt.plot(x1, y1['S1'], label='S1')
plt.plot(x1, y1['S2'], label='S2')
plt.plot(x1, y1['S3'], label='S3')
plt.plot(x1, y1['S4'], label='S4')
# plt.plot(x1, y1['B1'], label='B1')
# plt.plot(x1, y1['B2'], label='B2')
# plt.plot(x1, y1['F1'], label='F1')
# plt.plot(x1, y1['F2'], label='F2')
# plt.plot(x1, y1['F3'], label='F3')
# plt.plot(x1, y1['F4'], label='F4')
plt.figure()
plt.cla()

plt.plot(x2, y2['S1'], label='S1')
plt.plot(x2, y2['S2'], label='S2')
plt.plot(x2, y2['S3'], label='S3')
plt.plot(x2, y2['S4'], label='S4')
# plt.plot(x2, y2['B1'], label='B1')
# plt.plot(x2, y2['B2'], label='B2')
# plt.plot(x2, y2['F1'], label='F1')
# plt.plot(x2, y2['F2'], label='F2')
# plt.plot(x2, y2['F3'], label='F3')
# plt.plot(x2, y2['F4'], label='F4')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
