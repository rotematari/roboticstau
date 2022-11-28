import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ser = serial.Serial('/dev/ttyACM0', 115200)

count = 0
line = ser.readline()  # read a byte
string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
string = string.strip()
string.replace("'", '')
string.replace("[", '')
string.replace("]", '')

list_data = list(string.split(","))
list_data.remove('')
data = [float(i) for i in list_data]
x = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
y6 = []
y7 = []
y8 = []
y9 = []
data_len = len(data)


# print format: t,Gx,Gy,Gz,Ax,Ay,Az,Mx,My,Mz,P1,P2,P3,P4,P5,F1,F2,F3,F4,B1,B2,B3,S1,S2
def animate(i):
    line = ser.readline()  # read a byte
    string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
    string = string.strip()
    string.replace("'", '')
    string.replace("[", '')
    string.replace("]", '')
    list_data = list(string.split(","))
    list_data.remove('')
    data = [float(i) for i in list_data]  # convert  string to a float
    data_len = len(data)

    x.append(data[0])
    y1.append(data[data_len - 1])
    y2.append(data[data_len - 2])
    y3.append(data[data_len - 3])
    y4.append(data[data_len - 4])
    y5.append(data[data_len - 5])
    y6.append(data[data_len - 6])
    y7.append(data[data_len - 7])
    y8.append(data[data_len - 8])
    y9.append(data[data_len - 9])

    plt.cla()
    plt.plot(x, y1, label='S1')
    plt.plot(x, y2, label='S2')
    plt.plot(x, y3, label='B3')
    plt.plot(x, y4, label='B2')
    plt.plot(x, y5, label='B1')
    plt.plot(x, y6, label='F4')
    plt.plot(x, y7, label='F3')
    plt.plot(x, y8, label='F2')
    plt.plot(x, y9, label='F1')
    plt.legend(loc='upper left')
    plt.tight_layout()



ani = animation.FuncAnimation(plt.gcf(), animate, interval=1)
plt.tight_layout()
plt.show()
