from __future__ import print_function
import time as t
import sys
from os import path, getenv
from time import time, sleep
import numpy as np
import argparse

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
marker_data = []
natnet.run()

def record_data()
for i in range(10):
    marker_data.append(natnet.call())


natnet.stop()


