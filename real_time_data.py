import serial
import numpy as np
from torch.utils.data import Dataset
import torch

ser = serial.Serial('/dev/ttyACM0', 115200)


class Data(Dataset):
    def __init__(self):
        line = ser.readline()  # read a byte
        string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
        string = string.strip()
        string.replace("'", '')
        string.replace("[", '')
        string.replace("]", '')
        np_arr = np.fromstring(string, dtype=np.float32, sep=',')
        np_arr = np_arr[16:]
        self.x = torch.from_numpy(np_arr)
        self.n_samples = np_arr.shape
        print(self.n_samples)

    def __getitem__(self):
        return self.x

    def __len__(self):
        return self.n_samples
data = Data()