from os.path import join

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


dirpath = '/home/roblab15/Documents/FMG_project/data'
Title1 = 'state_0/state_0_1'
fileName1 = join(dirpath, Title1) + '.csv'
fileName2 = join(dirpath, Title1) + '.csv'

df1 = pd.read_csv(fileName1)
df2 = pd.read_csv(fileName2)

x = df1['time']
y = df2.drop(['time'], axis=1, inplace=False)


plt.cla()
plt.plot(x, y['S1'], label='S1')
plt.plot(x, y['S2'], label='S2')
plt.plot(x, y['S3'], label='S3')
plt.plot(x, y['S4'], label='S4')
plt.plot(x, y['B1'], label='B1')
plt.plot(x, y['B2'], label='B2')
plt.plot(x, y['F1'], label='F1')
plt.plot(x, y['F2'], label='F2')
plt.plot(x, y['F3'], label='F3')
plt.plot(x, y['F4'], label='F4')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
# def animate(i):
#     df = pd.read_csv(fileName)
#     # print(df)
#     x = df['time']
#     y1 = df['S1']
#     y2 = df['S2']
#     #y3 = df["B3"]
#     # y4 = df[20]
#     plt.cla()
#     plt.plot(x, y1, label='S1')
#     plt.plot(x, y2, label='S2')
#     #plt.plot(x, y3, label='B3')
#     #  plt.plot(x, y4, label='B2')
#     plt.legend(loc='upper left')
#     plt.tight_layout()
#
#
# ani = animation.FuncAnimation(plt.gcf(), animate, interval=1)
# plt.tight_layout()
# plt.show()
