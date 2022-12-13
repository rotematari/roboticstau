from os import listdir
from os.path import isfile, join
import serial

filepath = '//data'
dirpath = '/home/roblab15/Documents/FMG_project/data'

filesanddir = [f for f in listdir(dirpath)]
# make sure the 'COM#' is set according the Windows Device Manager
ser = serial.Serial('/dev/ttyACM0', 115200)
# print format: t,Gx,Gy,Gz,Ax,Ay,Az,Mx,My,Mz,F1,F2,F3,F4,B1,B2,S1,S2,S3,S4,class
# make new file names and locate them in the correct directory
for i in filesanddir:
    if not i == 'tests':
        filepath = dirpath + '/' + i
        onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
        print("move arm to :", i)
        num = str(len(onlyfiles))
        state = input("enter state( 0/1/2/3..ect :\n")
        print(state)
        print(onlyfiles)
        Title = i + '_' + num
        fileName = Title + '.csv'
        if fileName in onlyfiles:

            num = int(num) + 1
            # print(num)
            Title = i + '_' + str(num)
            fileName = Title + '.csv'
            print(fileName)
            f = open(join(filepath, fileName), "w")
            f.write("time,")
            f.write("Gx,Gy,Gz,Ax,Ay,Az,Mx,My,Mz,")
            # f.write("P1,P2,P3,P4,P5,")
            f.write("F1,F2,")
            f.write("B1,B2,")
            f.write("S1,S2,S3,S4,S5")
            f.write("class\n")
            for j in range(500):
                line = ser.readline()  # read a byte
                string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
                string = string.strip()
                string.replace("'", '')
                string.replace("[", '')
                string.replace("]", '')
                f.write(f'{string}' + f'{str(state)}' + '\n')
        else:
            Title = i + '_' + num
            fileName = Title + '.csv'
            print(fileName)
            f = open(join(filepath, fileName), "w")
            f.write("time,")
            f.write("Gx,Gy,Gz,Ax,Ay,Az,Mx,My,Mz,")
            # f.write("P1,P2,P3,P4,P5,")
            f.write("F1,F2,")
            f.write("B1,B2,")
            f.write("S1,S2,S3,S4,S5")
            f.write("class\n")
            for j in range(500):
                line = ser.readline()  # read a byte
                string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
                string = string.strip()
                string.replace("'", '')
                string.replace("[", '')
                string.replace("]", '')
                f.write(f'{string}' + f'{str(state)}' + '\n')

ser.close()
f.close()
print(fileName)



