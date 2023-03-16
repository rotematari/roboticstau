from os import listdir
from os.path import isfile, join
import serial
import time



# print format: Gx,Gy,Gz,Ax,Ay,Az,Mx,My,Mz,S1-48,sesion_time_stamp,
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

    for j in range(1000):
        line = ser.readline()  # read a byte
        string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
        string = string.strip()
        string.replace("'", '')
        string.replace("[", '')
        string.replace("]", '')
        f.write(f'{string}' + f'{str(state)}' + '\n')
        
    t_end = time.time()




ser.close()
print("finished")