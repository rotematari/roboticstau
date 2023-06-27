import sys
import argparse
import yaml

with open('/home/robotics20/Documents/rotem/new_code/config.yaml', 'r') as f:
    args = yaml.safe_load(f)



print(args)

parser = argparse.ArgumentParser()
# define your arguments here
parser.add_argument_group(args)
args = argparse.Namespace(**args)
print(args)

args = parser.parse_args(namespace=args)

print(args)

