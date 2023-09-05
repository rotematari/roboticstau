import os

# Get the current directory of the script being run
current_directory = os.path.dirname(os.path.realpath(__file__))

# Navigate up two directories
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

# Change the working directory
os.chdir(parent_directory)

# Now, any file or directory paths will be relative to this new working directory
