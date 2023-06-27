import time
from os import listdir
from os.path import join, isfile

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import paramaters



dirpath = paramaters.parameters.dirpath
# Title1 = join(dirpath, 'relaxed/relaxed_0')
# Title2 = join(dirpath, 'relaxed/relaxed_1')
# fileName1 = Title1 + '.csv'
# fileName2 = Title2 + '.csv'
#
# df1 = pd.read_csv(fileName1)
# df2 = pd.read_csv(fileName2)
# # print(df1)
# # print(df2)
# x1 = df1['time']
# y1 = df1.drop(['time'], axis=1, inplace=False)
#
# x2 = df2['time']
# y2 = df2.drop(['time'], axis=1, inplace=False)
#
#
# plt.figure(1)
# plt.cla()
# plt.plot(x1, y1['S1'], label='S1')
# plt.plot(x1, y1['S2'], label='S2')
# plt.plot(x1, y1['S3'], label='S3')
# plt.plot(x1, y1['S4'], label='S4')
# plt.plot(x1, y1['S5'], label='S5')
# plt.plot(x1, y1['S6'], label='S6')
# plt.plot(x1, y1['S7'], label='S7')
# plt.plot(x1, y1['S8'], label='S8')
# plt.plot(x1, y1['S9'], label='S9')
# plt.plot(x1, y1['S10'], label='S10')
# plt.plot(x1, y1['S11'], label='S11')
# plt.legend(loc='upper left')
# plt.tight_layout()
#
# plt.figure(2)
# plt.cla()
# plt.plot(x2, y2['S1'], label='S1')
# plt.plot(x2, y2['S2'], label='S2')
# plt.plot(x2, y2['S3'], label='S3')
# plt.plot(x2, y2['S4'], label='S4')
# plt.plot(x2, y2['S5'], label='S5')
# plt.plot(x2, y2['S6'], label='S6')
# plt.plot(x2, y2['S7'], label='S7')
# plt.plot(x2, y2['S8'], label='S8')
# plt.plot(x2, y2['S9'], label='S9')
# plt.plot(x2, y2['S10'], label='S10')
# plt.plot(x2, y2['S11'], label='S11')
# plt.legend(loc='upper left')
# plt.tight_layout()
#
#
# plt.show()
# plt.close()


filesanddir = [f for f in listdir(dirpath)]
for dir_name in filesanddir:
    filepath = dirpath + '/' + dir_name
    onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    if dir_name == 'relaxed':
        for file_name in onlyfiles:


            fileName = file_name
            df = pd.read_csv(join(filepath, file_name))

            x1 = df['time']
            y1 = df.drop(['time'], axis=1, inplace=False)

            plt.figure()
            plt.cla()
            plt.plot(x1, y1['S1'], label='S1')
            plt.plot(x1, y1['S2'], label='S2')
            plt.plot(x1, y1['S3'], label='S3')
            plt.plot(x1, y1['S4'], label='S4')
            plt.plot(x1, y1['S5'], label='S5')
            plt.plot(x1, y1['S6'], label='S6')
            plt.plot(x1, y1['S7'], label='S7')
            plt.plot(x1, y1['S8'], label='S8')
            plt.plot(x1, y1['S9'], label='S9')
            plt.plot(x1, y1['S10'], label='S10')
            plt.plot(x1, y1['S11'], label='S11')
            plt.legend(loc='upper left')
            plt.tight_layout()


plt.show()
plt.close()