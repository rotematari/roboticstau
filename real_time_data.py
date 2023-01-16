import pandas as pd
import serial
import numpy as np
from torch.utils.data import Dataset
import torch
import data_agmuntation
import data_loader
import time
from sklearn.preprocessing import StandardScaler

ser = serial.Serial('/dev/ttyACM0', 115200)
items = data_loader.items


class Data(Dataset):
    def __init__(self, mean_relaxed, full_arr_std, calibrate=True, s=(200, 6)):
        relaxed_arr = np.zeros(s)
        full_arr = np.zeros((s[0] * 4, s[1]))

        if calibrate:
            if relaxed_arr[0, 0] == 0:
                # input("calibrate system press enter when in relaxed possision\n")
                print("calibrate system press enter when in relaxed possision\n")
                time.sleep(4)

                for i in range(len(relaxed_arr)):
                    line = ser.readline()  # read a byte
                    string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
                    string = string.strip()
                    string.replace("'", '')
                    string.replace("[", '')
                    string.replace("]", '')
                    np_arr1 = np.fromstring(string, dtype=np.float32, sep=',')
                    relaxed_arr[i] = np_arr1[12:18]
                    full_arr[i] = np_arr1[12:18]

                full_count: int = i
                i = 0
                print("calibrate system press enter when in forword possision\n")
                time.sleep(4)
                for i in range(len(relaxed_arr)):
                    line = ser.readline()  # read a byte
                    string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
                    string = string.strip()
                    string.replace("'", '')
                    string.replace("[", '')
                    string.replace("]", '')
                    np_arr1 = np.fromstring(string, dtype=np.float32, sep=',')

                    full_arr[full_count + i] = np_arr1[12:18]

                full_count = full_count + i
                i = 0
                print("calibrate system press enter when in left possision\n")
                time.sleep(4)
                for i in range(len(relaxed_arr)):
                    line = ser.readline()  # read a byte
                    string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
                    string = string.strip()
                    string.replace("'", '')
                    string.replace("[", '')
                    string.replace("]", '')
                    np_arr1 = np.fromstring(string, dtype=np.float32, sep=',')

                    full_arr[full_count + i] = np_arr1[12:18]

                full_count = full_count + i
                i = 0
                print("calibrate system press enter when in up possision\n")
                time.sleep(4)
                for i in range(len(relaxed_arr)):

                    line = ser.readline()  # read a byte
                    string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
                    string = string.strip()
                    string.replace("'", '')
                    string.replace("[", '')
                    string.replace("]", '')
                    np_arr1 = np.fromstring(string, dtype=np.float32, sep=',')

                    full_arr[full_count + i] = np_arr1[12:18]

            relaxed_data = pd.DataFrame(relaxed_arr)
            mean_relaxed = relaxed_data.mean()

            full_arr = pd.DataFrame(full_arr, columns=items)
            scaler = StandardScaler(with_mean=False)
            scaler.fit(full_arr)
            full_arr = scaler.transform(full_arr)
            full_arr_std = scaler.scale_

        s = (1, 6)
        window = 50
        np_arr = np.zeros(s)
        batch_arr = np.zeros(s)
        for i in range(window):
            # for j in range(len(np_arr)):
            line = ser.readline()  # read a byte
            string = line.decode('utf-8')  # ('latin-1')  # convert the byte string to a unicode string
            string = string.strip()
            string.replace("'", '')
            string.replace("[", '')
            string.replace("]", '')

            np_arr2 = np.fromstring(string, dtype=np.float32, sep=',')
            np_arr = np_arr2[12:18]

            # bais
            np_arr -= mean_relaxed

            #std
            np_arr /= full_arr_std
            np_arr = np.array(np_arr)
            np_arr = np.expand_dims(np_arr, axis=0)
            # collect all window
            if i == 0:
                batch_arr = np_arr
            else:

                # batch_arr = np.expand_dims(batch_arr, axis=0)
                # np_arr = np.expand_dims(np_arr, axis=0)
                batch_arr = np.concatenate((batch_arr, np_arr), axis=0)




        # batch_arr = np.array(batch_arr)
        batch_arr_mean = batch_arr.mean(axis=0)
        batch_arr_std = batch_arr.std(axis=0)

        real_time_data = np.concatenate((batch_arr_mean, batch_arr_std))
        real_time_data = np.array(real_time_data, dtype=np.float32)
        real_time_data = np.expand_dims(real_time_data,axis=0)
        # np_arr = data_agmuntation.stdnorm(np_arr)
        self.mean_relaxed = mean_relaxed
        self.full_arr_std = full_arr_std
        self.x = torch.from_numpy(real_time_data)
        self.n_samples = np_arr.shape
        # self.y = torch.from_numpy(np.array([1], dtype=np.float32))

    def __getitem__(self):
        return self.x, self.y

    def __len__(self):
        return self.n_samples

