import serial
import numpy as np
from torch.utils.data import Dataset
import torch

ser = serial.Serial('/dev/ttyACM1', 115200)


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
        print(np_arr)
        np_arr = np.array([np_arr])
        print(np_arr)
        self.x = torch.from_numpy(np_arr)
        self.n_samples = np_arr.shape
        self.y = torch.from_numpy(np.array([1], dtype=np.float32))


    def __getitem__(self):
        return self.x, self.y

    def __len__(self):
        return self.n_samples
data = Data()