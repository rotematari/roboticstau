from os import listdir
from os.path import isfile, join
import numpy as np
import serial
import time as t
# import NatNet client
from NatNetClient import NatNetClient
#import natnetclient as natnet

dirpath = '/home/robotics20/Documents/rotem/data'

# dirs = [f for f in listdir(dirpath)]
# make sure the 'COM#' is set according the Windows Device Manager
# ser = serial.Serial('/dev/ttyACM0', 115200)


# print format: t,Gx,Gy,Gz,Ax,Ay,Az,Mx,My,Mz,F1,F2,F3,F4,B1,B2,S1,S2,S3,S4,class
# make new file names and locate them in the correct directory
def write_first_line(f):
    # f.write("time,")
    # f.write("Gx,Gy,Gz,Ax,Ay,Az,Mx,My,Mz,")
    f.write("S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18, S19, S20,")
    f.write("S21, S22, S23, S24, S25, S26, S27, S28, S29, S30, S31, S32, S33, S34, S35, S36, S37, S38, S39, S40, S41, S42, S43, S44, S45, S46, S47, S48,")
    f.write("M1x,M1y,M1z,M2x,M2y,M2z,M3x,M3y,M3z,M4x,M4y,M4z,")
    f.write("sesion_time_stamp,\n")


def receiveNewFrame( frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount,
                    labeledMarkerCount, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged ):
    print( "Received frame", frameNumber )

# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receiveRigidBodyFrame( id, position, rotation ):
    print( "Received frame for rigid body", id )

def receiveRigidBodyList( rigidBodyList, stamp ):
    for (ac_id, pos, quat, valid) in rigidBodyList:
        # print("rigidBodyList")
        # print(rigidBodyList)
        # print(type(rigidBodyList))
        # print(len(rigidBodyList))
        if not valid:
            # skip if rigid body is not valid
            continue
        
        # print('id: ', ac_id, 'pos:', pos, 'quat:', quat) 


def init_natnetClient():
           
    # start natnet interface
    natnet = NatNetClient(rigidBodyListListener=receiveRigidBodyList,server="132.66.51.232")#rigidBodyListListener=receiveRigidBodyList)

    keys = ['chest', 'shoulder', 'elbow', 'wrist']
    chest = 1
    shoulder = 2
    elbow = 3
    wrist = 4
    return natnet

    # This dictionary matches the rigid body id (key) to it's name (value)
    motive_matcher = {chest: 'chest',
                        shoulder: 'shoulder',
                        shoulder: 'elbow',
                        elbow: 'wrist',}


def write_line(f,marker_data):

    sesion_time_stamp = t.strftime("%d_%b_%Y_%H:%M", t.gmtime())


    # line = ser.readline()  # read a byte
    # sensor_string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
    # sensor_string = sensor_string.strip()
    # sensor_string.replace("'", '')
    # sensor_string.replace("[", '')
    # sensor_string.replace("]", '')


     # test 
    sensor_string = ''.join(str(i)+',' for i in range(48))

    print(marker_data)
    marker_string =[]
    for i in range(len(marker_data)):

        marker_string += [str(j)for j in marker_data[i][1]] 

    marker_string = ''.join(str(s)+',' for s in marker_string)

    # sensor_string , marker_string , sesion_time_stamp
    f.write(f'{sensor_string}' +f'{marker_string}'+ f'{sesion_time_stamp}' + '\n')
    






if __name__ == '__main__':
    
    t_start = t.time()
    NatNet = init_natnetClient()
    # print(fileName)
    f = open(join('new_code/data/data', 'test.csv'), "w")

    write_first_line(f)
    NatNet.run()
    for i in range(20):
      marker_data = NatNet.rigidBodyList

      write_line(f,marker_data=marker_data)
    
    f.close()
    NatNet.stop()

    t_end = t.time()
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

# ser.close()
# print("finished")