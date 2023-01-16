from os import listdir
from os.path import isfile, join
import serial
import time

filepath = '//data'
dirpath = '/home/roblab20/Documents/rotem/data'

dirs = [f for f in listdir(dirpath)]
# make sure the 'COM#' is set according the Windows Device Manager
ser = serial.Serial('/dev/ttyACM0', 115200)


# print format: t,Gx,Gy,Gz,Ax,Ay,Az,Mx,My,Mz,F1,F2,F3,F4,B1,B2,S1,S2,S3,S4,class
# make new file names and locate them in the correct directory
def write_firs_line(f):
    f.write("time,")
    f.write("Gx,Gy,Gz,Ax,Ay,Az,Mx,My,Mz,")
    f.write("S1,S2,")
    f.write("S3,S4,")
    f.write("S5,S6,S7,S8,S9,S10,S11,")
    f.write("class\n")


def write_line(f, state=None):
    t_start = time.time()

    for j in range(2000):
        line = ser.readline()  # read a byte
        string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
        string = string.strip()
        string.replace("'", '')
        string.replace("[", '')
        string.replace("]", '')
        f.write(f'{string}' + f'{str(state)}' + '\n')
    t_end = time.time()

    print(t_end-t_start)
for i in range(len(dirs)):

    state = input("choose state : 0=relaxed, 1=forward , 2=left , 3=up \n")

    if state == '0':
        print('ok')
        filepath = dirpath + '/' + 'relaxed'
        onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
        num = str(len(onlyfiles))
        num = int(num)
        Title = 'relaxed_' + str(num)
        fileName = Title + '.csv'
        if fileName in onlyfiles:
            num = int(num) + 1
            Title = 'relaxed_' + str(num)
            fileName = Title + '.csv'

        print(fileName)
        f = open(join(filepath, fileName), "w")
        write_firs_line(f)
        write_line(f, state=state)
        f.close()
    elif state == '1':
        filepath = dirpath + '/' + 'forward'
        onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
        num = str(len(onlyfiles))
        num = int(num)
        Title = 'forward_' + str(num)
        fileName = Title + '.csv'
        if fileName in onlyfiles:
            num = int(num) + 1
            Title = 'forward_' + str(num)
            fileName = Title + '.csv'
        print(fileName)
        f = open(join(filepath, fileName), "w")
        write_firs_line(f)
        write_line(f, state=state)
        f.close()
    elif state == '2':
        filepath = dirpath + '/' + 'left'
        onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
        num = str(len(onlyfiles))
        num = int(num)
        Title = 'left_' + str(num)
        fileName = Title + '.csv'
        if fileName in onlyfiles:
            num = int(num) + 1
            Title = 'left_' + str(num)
            fileName = Title + '.csv'
        print(fileName)
        f = open(join(filepath, fileName), "w")
        write_firs_line(f)
        write_line(f, state=state)
        f.close()
    elif state == '3':
        filepath = dirpath + '/' + 'up'
        onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
        num = str(len(onlyfiles))
        num = int(num)
        Title = 'up_' + str(num)
        fileName = Title + '.csv'
        if fileName in onlyfiles:
            num = int(num) + 1
            Title = 'up_' + str(num)
            fileName = Title + '.csv'
        print(fileName)
        f = open(join(filepath, fileName), "w")
        write_firs_line(f)
        write_line(f, state=state)
        f.close()

ser.close()
print("finished")
