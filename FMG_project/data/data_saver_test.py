from os import listdir
from os.path import isfile, join
import os 
# Get the current directory of the script being run
current_directory = os.path.dirname(os.path.realpath(__file__))

# Navigate up  directori
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
# Change the working directory
os.chdir(parent_directory)

import numpy as np
import serial
import time as t

from NatNetClient import NatNetClient

import matplotlib.pyplot as plt 
import yaml
import argparse
import pandas as pd


data_dir = r'data/data_test'

with open(r'config.yaml', 'r') as f:
    args = yaml.safe_load(f)


    config = argparse.Namespace(**args)

def receiveRigidBodyList(rigidBodyList, timestamp):
    for (ac_id, pos, quat, valid) in rigidBodyList:
        if not valid:
            continue


def init_natnetClient():
           
    # start natnet interface
    natnet = NatNetClient(rigidBodyListListener=receiveRigidBodyList,server="132.66.51.232")#rigidBodyListListener=receiveRigidBodyList)


    return natnet


if __name__ == '__main__':
    print("make sure the assets set to xyz")
    keys = ['chest', 'shoulder', 'elbow', 'wrist']
    chest = 1
    shoulder = 2
    elbow = 3
    wrist = 4
    motive_matcher = {chest: 'chest',
                      shoulder: 'shoulder',
                      elbow: 'elbow',
                      wrist: 'wrist',
                      }

    natnet = init_natnetClient()
    natnet.run()
    # make sure the assets set to xyz
    locations = {
        'chest':[],
        'shoulder':[],
        'elbow':[],
        'wrist':[],
    }
    labels = "MCx,MCy,MCz,MSx,MSy,MSz,MEx,MEy,MEz,MWx,MWy,MWz"

    rigid_body = natnet.rigidBodyList
    if not os.path.exists(r'./data/data_labels/'):
                os.mkdir(r'./data/data_labels/')

    with open('./data/data_labels/labels.csv','w') as f:
        f.write(labels +'\n')

        for i in range(100):

            for j in range(len(rigid_body)):
                locations[motive_matcher[rigid_body[j][0]]].append(rigid_body[j][1])
            # read serial line
            locations_string = ','.join(map(str,locations['chest'][0]+locations['shoulder'][0]+locations['elbow'][0]+locations['wrist'][0]))
            
            

            f.write(locations_string + '\n')



        print(locations_string)

