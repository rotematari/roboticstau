import sys
import argparse
import yaml

with open('/home/robotics20/Documents/rotem/new_code/tests/test_config.yaml', 'r') as f:
    args = yaml.safe_load(f)

print

parser = argparse.ArgumentParser()
# define your arguments here

args = argparse.Namespace(**args)

args = parser.parse_args(namespace=args)


