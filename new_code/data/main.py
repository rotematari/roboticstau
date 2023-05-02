from __future__ import print_function
import time as t
import sys
from os import path, getenv,listdir
from time import time, sleep
import numpy as np
import argparse
from os import listdir
from os.path import isfile, join
import serial
import data_saver
# import NatNet client
from NatNetClient import NatNetClient
#import natnetclient as natnet



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

    # This dictionary matches the rigid body id (key) to it's name (value)
    motive_matcher = {chest: 'chest',
                        shoulder: 'shoulder',
                        shoulder: 'elbow',
                        elbow: 'wrist',}



def record_data(NatNetClient,num_points):

    start_time = t.time()
    #make sure mitive outs x,y,z
    full_data = [] 
    marker_data = []
    NatNetClient.run()
    for i in range(num_points):
        marker_data.append(NatNetClient.call())





    total_time =t.time() -start_time
    NatNetClient.stop()
    print('duration for:',num_points,'points = ', total_time)


    return marker_data,sensor_data, time_stamp



