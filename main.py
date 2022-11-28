import serial
import time
import matplotlib.pyplot as plt
import xlsxwriter.table as xlsx
import numpy as np
import matplotlib.animation as animation



Title = 'state_9_t1'
fileName = Title + '.csv'

# G1 = []
# G2 = []
# G3 = []
# A1 = []
# A2 = []
# A3 = []
# M1 = []
# M2 = []
# M3 = []
# P1 = []
# P2 = []
# P3 = []
# P4 = []
# P5 = []
# F1 = []
# F2 = []
# F3 = []
# F4 = []
# B1 = []
# B2 = []
# B3 = []
# S1 = []
# S2 = []

# timeInSeconds = []
# timeToCSV = 0
# shoulderToFileData = ''
# GyroToFileData = ''
# AccelToFileData = ''
# magToFileData = ''
# palmToFileData = ''
# forearmToFileData = ''
# bicepToFileData = ''
# shoulderToFileData = ''
# timeToCSV = ''

# make sure the 'COM#' is set according the Windows Device Manager
ser = serial.Serial('/dev/ttyACM0', 115200)


f = open(fileName, "w")
# f = open("testFile.CSV.CSV", "w")
f.write("time,")
f.write("Gx,Gy,Gz,Ax,Ay,Az,Mx,My,Mz,")
f.write("P1,P2,P3,P4,P5,")
f.write("F1,F2,F3,F4,")
f.write("B1,B2,B3,")
f.write("S1,S2,\n")


# print format: t,Gx,Gy,Gz,Ax,Ay,Az,Mx,My,Mz,P1,P2,P3,P4,P5,F1,F2,F3,F4,B1,B2,B3,S1,S2
time0 = time.time()
for i in range(5000):
    line = ser.readline()  # read a byte
    string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
    string = string.strip()
    string.replace("'", '')
    string.replace("[", '')
    string.replace("]", '')
    f.write(f'{string}' + '\n')
    print(str(string))

time1 = time.time()
print(time1-time0)
# for i in range(50):
#     print(i)
#     line = ser.readline()   # read a byte
#     if line:
#         string = line.decode("utf-8")  # convert the byte string to a unicode string
#         string = string.strip()
#         tempData = string.split(",")
#         print(tempData)
#         if tempData[0] == "G":
#             G1.append(int(tempData[1]))
#             G2.append(int(tempData[2]))
#             G3.append(int(tempData[3]))
#             GyroToFileData = tempData[1] + ',' + tempData[2] + ',' + tempData[3]
#             f.write(str(GyroToFileData) + ',')
#         if tempData[0] == "A":
#             A1.append(int(tempData[1]))
#             A2.append(int(tempData[2]))
#             A3.append(int(tempData[3]))
#             AccelToFileData = tempData[1] + ',' + tempData[2] + ',' + tempData[3]
#             f.write(str(AccelToFileData) + ',')
#         if tempData[0] == "M":
#             M1.append(int(tempData[1]))
#             M2.append(int(tempData[2]))
#             M3.append(int(tempData[3]))
#             magToFileData = tempData[1] + ',' + tempData[2] + ',' + tempData[3]
#             f.write(str(magToFileData) + ',')
#         if tempData[0] == "P":
#             P1.append(int(tempData[1]))
#             P2.append(int(tempData[2]))
#             P3.append(int(tempData[3]))
#             P4.append(int(tempData[4]))
#             P5.append(int(tempData[5]))
#             palmToFileData = tempData[1] + ',' + tempData[2] + ',' + tempData[3] + ',' + tempData[4] + ',' + tempData[5]
#             f.write(str(palmToFileData) + ',')
#         if tempData[0] == "F":
#             F1.append(int(tempData[1]))
#             F2.append(int(tempData[2]))
#             F3.append(int(tempData[3]))
#             F4.append(int(tempData[4]))
#             forearmToFileData = tempData[1] + ',' + tempData[2] + ',' + tempData[3] + ',' + tempData[4]
#             f.write(str(forearmToFileData) + ',')
#         if tempData[0] == "B":
#             B1.append(int(tempData[1]))
#             B2.append(int(tempData[2]))
#             B3.append(int(tempData[3]))
#             bicepToFileData = tempData[1] + ',' + tempData[2] + ',' + tempData[3]
#             f.write(str(bicepToFileData) + ',')
#         if tempData[0] == "S":
#             S1.append(int(tempData[1]))
#             S2.append(int(tempData[2]))
#             shoulderToFileData = tempData[1] + ',' + tempData[2] + ',' + tempData[3]
#             f.write(str(shoulderToFileData) + ',')
#         if tempData[0] == "t":
#             timeInSeconds.append(int(tempData[1])/1000)
#             timeToCSV = tempData[1]
#             f.write(str(timeToCSV))
#
#         f.write('\n')
#         print(str(GyroToFileData) + ',' + str(AccelToFileData) + ',' + str(magToFileData) + ',' +
#               str(palmToFileData) + ',' + str(forearmToFileData) + ',' + str(bicepToFileData) + ',' +
#               str(shoulderToFileData) + ',' + str(timeToCSV))


# print(tempData)
ser.close()
f.close()

# print(len(timeInSeconds))
# print(len(P1))
# print(len(P2))
# print(len(P3))
# print(len(P4))
# print(len(P5))
# print(G1)
# print(G2)
# print(G3)
# print(A1)
# print(A2)
# print(A3)
# print(M1)
# print(M2)
# print(M3)
# print(P1)
# print(P2)
# print(P3)
# print(P4)
# print(P5)
# print(F1)
# print(F2)
# print(F3)
# print(F4)
# print(B1)
# print(B2)
# print(B3)
# print(S1)
# print(S2)


# plt.plot(timeInSeconds, P1[0:len(P1-1)])
# plt.plot(timeInSeconds, P2[0:len(P2-1)])
# plt.plot(timeInSeconds, P3[0:len(P3-1)])
# plt.plot(timeInSeconds, P4[0:len(P4-1)])
# plt.plot(timeInSeconds, P5[0:len(P5-1)])

# plt.plot(timeInSeconds, S1)
# plt.plot(timeInSeconds, S2)
# plt.plot(timeInSeconds, B3)


# plt.show()
