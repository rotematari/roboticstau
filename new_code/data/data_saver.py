from os import listdir
from os.path import isfile, join
import serial
import time


dirpath = '/home/robotics20/Documents/rotem/data'

dirs = [f for f in listdir(dirpath)]
# make sure the 'COM#' is set according the Windows Device Manager
ser = serial.Serial('/dev/ttyACM0', 115200)


# print format: t,Gx,Gy,Gz,Ax,Ay,Az,Mx,My,Mz,F1,F2,F3,F4,B1,B2,S1,S2,S3,S4,class
# make new file names and locate them in the correct directory
def write_first_line(f):
    f.write("time,")
    f.write("Gx,Gy,Gz,Ax,Ay,Az,Mx,My,Mz,")
    f.write("S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18, S19, S20,")
    f.write("S21, S22, S23, S24, S25, S26, S27, S28, S29, S30, S31, S32, S33, S34, S35, S36, S37, S38, S39, S40, S41, S42, S43, S44, S45, S46, S47, S48,")
    f.write("M1x,M1y,M1z,M2x,M2y,M2z,M3x,M3y,M3z,M4x,M4y,M4z")
    f.write("sesion_time_stamp\n")


def write_line(f, state=None):

    sesion_time_stamp = time.strftime("%d_%b_%Y_%H:%M", time.gmtime())
    t_start = time.time()

    for j in range(1000):

        line = ser.readline()  # read a byte
        string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
        string = string.strip()
        string.replace("'", '')
        string.replace("[", '')
        string.replace("]", '')



        
        f.write(f'{string}' + f'{sesion_time_stamp}' + '\n')
    
    
    t_end = time.time()
    print(t_end-t_start)
# for i in range(len(dirs)):

#     state = input("choose state : 0=relaxed, 1=forward , 2=left , 3=up \n")

#     if state == '0':
#         print('ok')
#         filepath = dirpath + '/' + 'relaxed'
#         onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
#         num = str(len(onlyfiles))
#         num = int(num)
#         Title = 'relaxed_' + str(num)
#         fileName = Title + '.csv'
#         if fileName in onlyfiles:
#             num = int(num) + 1
#             Title = 'relaxed_' + str(num)
#             fileName = Title + '.csv'

#         print(fileName)
#         f = open(join(filepath, fileName), "w")
#         write_firs_line(f)
#         write_line(f, state=state)
#         f.close()

#     elif state == '1':
#         filepath = dirpath + '/' + 'forward'
#         onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
#         num = str(len(onlyfiles))
#         num = int(num)
#         Title = 'forward_' + str(num)
#         fileName = Title + '.csv'
#         if fileName in onlyfiles:
#             num = int(num) + 1
#             Title = 'forward_' + str(num)
#             fileName = Title + '.csv'
#         print(fileName)
#         f = open(join(filepath, fileName), "w")
#         write_firs_line(f)
#         write_line(f, state=state)
#         f.close()
#     elif state == '2':
#         filepath = dirpath + '/' + 'left'
#         onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
#         num = str(len(onlyfiles))
#         num = int(num)
#         Title = 'left_' + str(num)
#         fileName = Title + '.csv'
#         if fileName in onlyfiles:
#             num = int(num) + 1
#             Title = 'left_' + str(num)
#             fileName = Title + '.csv'
#         print(fileName)
#         f = open(join(filepath, fileName), "w")
#         write_firs_line(f)
#         write_line(f, state=state)
#         f.close()
#     elif state == '3':
#         filepath = dirpath + '/' + 'up'
#         onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
#         num = str(len(onlyfiles))
#         num = int(num)
#         Title = 'up_' + str(num)
#         fileName = Title + '.csv'
#         if fileName in onlyfiles:
#             num = int(num) + 1
#             Title = 'up_' + str(num)
#             fileName = Title + '.csv'
#         print(fileName)
#         f = open(join(filepath, fileName), "w")
#         write_firs_line(f)
#         write_line(f, state=state)
#         f.close()

ser.close()
print("finished")