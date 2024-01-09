from os import listdir
from os.path import isfile, join
import os 
import yaml
import argparse
import pandas as pd

from NatNetClient import NatNetClient

# Get the current directory of the script being run
current_directory = os.path.dirname(os.path.realpath(__file__))

# Navigate up  directori
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
# Change the working directory
os.chdir(parent_directory)

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

def read_sample(config,natnet):


    locations = config.locations

    motive_matcher = config.motive_matcher
    #warmup
    for i in range(3):
        natnet.rigidBodyList

    rigid_bodys = natnet.rigidBodyList

    for j in range(len(rigid_bodys)):
        locations[motive_matcher[rigid_bodys[j][0]]].append(rigid_bodys[j][1])

    return locations

def main(config):

    natnet = init_natnetClient()

    natnet.run()


    locations = read_sample(config,natnet)

    print(locations)


if __name__=="__main__":

    main(config)