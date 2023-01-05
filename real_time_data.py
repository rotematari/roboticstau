import pandas as pd
import serial
import numpy as np
from torch.utils.data import Dataset
import torch
import data_agmuntation
import data_loader
ser = serial.Serial('/dev/ttyACM0', 115200)
items = data_loader.items



class Data(Dataset):
    def __init__(self, mean_relaxed, calibrate=True, s=(200, 6)):
        relaxed_arr = np.zeros(s)
        if calibrate:
            if relaxed_arr[0, 0] == 0:
                input("calibrate system press enter when in relaxed possision\n")
                for i in range(len(relaxed_arr)):
                    line = ser.readline()  # read a byte
                    string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
                    string = string.strip()
                    string.replace("'", '')
                    string.replace("[", '')
                    string.replace("]", '')
                    np_arr1 = np.fromstring(string, dtype=np.float32, sep=',')
                    relaxed_arr[i] = np_arr1[12:18]

            relaxed_data = pd.DataFrame(relaxed_arr)
            mean_relaxed = relaxed_data.mean()

        s = (1, 6)
        np_arr = np.zeros(s)

        for j in range(len(np_arr)):
            line = ser.readline()  # read a byte
            string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
            string = string.strip()
            string.replace("'", '')
            string.replace("[", '')
            string.replace("]", '')

            np_arr2 = np.fromstring(string, dtype=np.float32, sep=',')
            np_arr[j] = np_arr2[12:18]



        for i in range(len(np_arr[0])):
            np_arr[0, i] -= mean_relaxed[i]

        # np_arr = data_agmuntation.stdnorm(np_arr)
        self.mean_relaxed = mean_relaxed
        self.x = torch.from_numpy(np.array(np_arr, dtype=np.float32))
        self.n_samples = np_arr.shape
        self.y = torch.from_numpy(np.array([1], dtype=np.float32))


    def __getitem__(self):
        return self.x, self.y

    def __len__(self):
        return self.n_samples




# data = Data()
